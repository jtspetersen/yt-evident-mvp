# app/pipeline/consolidate_claims.py
"""
Consolidation stage: deduplicates overlapping claims and groups
contextually related claims into narrative clusters.
Runs between extract_claims and review_claims.
"""
import json
import os
import sys
from typing import List, Tuple, Optional, Callable

from app.tools.ollama_client import ollama_chat
from app.tools.json_extract import extract_json
from app.schemas.claim import Claim
from app.schemas.claim_group import ClaimGroup


def _should_log(level: str) -> bool:
    levels = {"DEBUG": 0, "INFO": 1, "WARNING": 2, "ERROR": 3}
    current_level = os.environ.get("EVIDENT_LOG_LEVEL", "INFO")
    return levels.get(level, 1) >= levels.get(current_level, 1)


SYSTEM = """You are an analytical claim consolidation system. You will receive a list of factual claims extracted from a video transcript, plus transcript context.

Your TWO tasks:

## Task 1 — Deduplicate
Identify claims that say the same thing (from overlapping extraction chunks). For each duplicate pair, decide which version is better written (more complete, more precise) and mark the other for removal.

Rules:
- Only mark TRUE duplicates — claims making the same factual assertion
- If two claims cover similar territory but make distinct points, keep both
- Prefer the version with more context, better specificity, or a more precise quote

## Task 2 — Narrative Grouping
Identify clusters of claims that work together to build a larger narrative argument. A narrative group is 2+ claims that:
- Build toward a shared thesis or conclusion
- Are presented by the speaker as connected evidence for a larger point
- Would be misleading if evaluated individually but reveal a pattern when viewed together

For each group, write a 1-2 sentence narrative_thesis summarizing the argument the speaker is constructing, and identify the rhetorical_strategy being used.

Common rhetorical strategies:
- "cherry_picking" — selecting only facts that support one conclusion
- "gish_gallop" — overwhelming with many claims to imply a pattern
- "false_causation" — implying A causes B without evidence
- "emotional_framing" — using emotionally charged examples to drive a conclusion
- "motte_and_bailey" — mixing defensible claims with indefensible ones
- "appeal_to_authority" — citing studies/experts selectively
- "anecdote_as_evidence" — using individual stories as proof of systemic claims

Rules:
- A claim can belong to at most ONE group
- Not every claim needs a group — standalone claims stay ungrouped
- Groups must have at least 2 claims
- The narrative_thesis should describe what the speaker is arguing, not just a topic

## Output Format

Return ONLY a JSON object with this exact structure:
{
  "duplicates": [
    {"keep": "C009", "drop": "C012", "reason": "Same claim about LA rent drop; C009 is more complete"}
  ],
  "groups": [
    {
      "group_id": "G001",
      "narrative_thesis": "The speaker argues that illegal immigration has created an enormous fiscal burden, citing a specific study to claim each immigrant costs $68,000",
      "claim_ids": ["C015", "C016", "C018"],
      "rhetorical_strategy": "appeal_to_authority"
    }
  ]
}

If no duplicates are found, return "duplicates": [].
If no narrative groups are found, return "groups": []."""


def _build_user_prompt(claims: List[Claim], transcript_json: dict) -> str:
    """Build the user prompt with claims and transcript context."""
    # Claims as compact JSON
    claims_data = []
    for c in claims:
        claims_data.append({
            "claim_id": c.claim_id,
            "segment_id": c.segment_id,
            "timestamp": c.timestamp,
            "claim_text": c.claim_text,
            "quote_from_transcript": c.quote_from_transcript,
            "claim_type": c.claim_type,
        })

    # Include first 30 transcript segments for context
    segments = transcript_json.get("segments", [])[:30]
    context_segments = []
    for s in segments:
        context_segments.append({
            "id": s.get("id"),
            "timestamp": s.get("timestamp"),
            "text": s.get("text", ""),
        })

    prompt = (
        f"## Claims to consolidate ({len(claims_data)} total)\n\n"
        f"{json.dumps(claims_data, indent=2, ensure_ascii=False)}\n\n"
        f"## Transcript context (first {len(context_segments)} segments)\n\n"
        f"{json.dumps(context_segments, indent=2, ensure_ascii=False)}\n\n"
        "Analyze these claims. Identify duplicates and narrative groups. "
        "Return ONLY the JSON object as specified."
    )
    return prompt


def _apply_consolidation(
    claims: List[Claim],
    raw_result: dict,
) -> Tuple[List[Claim], List[ClaimGroup]]:
    """
    Apply dedup + grouping results to the claim list.
    Returns (filtered_claims, groups).
    """
    # --- Deduplication ---
    drop_ids = set()
    duplicates = raw_result.get("duplicates", [])
    for dup in duplicates:
        drop_id = dup.get("drop")
        keep_id = dup.get("keep")
        if drop_id and keep_id:
            # Only drop if the keep_id actually exists
            claim_ids = {c.claim_id for c in claims}
            if keep_id in claim_ids and drop_id in claim_ids:
                drop_ids.add(drop_id)
                if _should_log("INFO"):
                    print(f"INFO: Dropping duplicate {drop_id} (keeping {keep_id}): {dup.get('reason', '')}", file=sys.stderr)

    # Filter out dropped claims
    filtered = [c for c in claims if c.claim_id not in drop_ids]

    # --- Build valid claim ID set after dedup ---
    valid_ids = {c.claim_id for c in filtered}

    # --- Narrative Grouping ---
    groups = []
    raw_groups = raw_result.get("groups", [])
    assigned_ids = set()  # Track claims already assigned to a group

    for rg in raw_groups:
        group_id = rg.get("group_id", "")
        thesis = rg.get("narrative_thesis", "")
        raw_claim_ids = rg.get("claim_ids", [])
        strategy = rg.get("rhetorical_strategy")

        if not group_id or not thesis:
            continue

        # Filter to only valid, unassigned claim IDs
        valid_member_ids = [
            cid for cid in raw_claim_ids
            if cid in valid_ids and cid not in assigned_ids
        ]

        # Need at least 2 members for a group
        if len(valid_member_ids) < 2:
            if _should_log("DEBUG"):
                print(f"DEBUG: Dissolving group {group_id} — only {len(valid_member_ids)} valid members", file=sys.stderr)
            continue

        group = ClaimGroup(
            group_id=group_id,
            narrative_thesis=thesis,
            claim_ids=valid_member_ids,
            rhetorical_strategy=strategy,
        )
        groups.append(group)
        assigned_ids.update(valid_member_ids)

    # Set group_id on member claims
    group_lookup = {}
    for g in groups:
        for cid in g.claim_ids:
            group_lookup[cid] = g.group_id

    updated_claims = []
    for c in filtered:
        if c.claim_id in group_lookup:
            c = c.model_copy(update={"group_id": group_lookup[c.claim_id]})
        updated_claims.append(c)

    # Renumber claims sequentially
    for i, c in enumerate(updated_claims, start=1):
        new_id = f"C{i:03d}"
        old_id = c.claim_id
        if old_id != new_id:
            # Update group references
            for g in groups:
                g.claim_ids = [new_id if cid == old_id else cid for cid in g.claim_ids]
            updated_claims[i - 1] = c.model_copy(update={"claim_id": new_id})

    return updated_claims, groups


def consolidate_claims(
    ollama_base: str,
    model: str,
    claims: List[Claim],
    transcript_json: dict,
    temperature: float = 0.1,
    progress_callback: Optional[Callable] = None,
) -> Tuple[List[Claim], List[ClaimGroup]]:
    """
    Consolidate extracted claims: remove duplicates and group related
    claims into narrative clusters.

    Returns (consolidated_claims, groups).
    If the LLM call fails, returns (original claims, []) gracefully.
    """
    def _cb(data):
        if progress_callback:
            progress_callback(data)

    # Skip if too few claims to consolidate
    if len(claims) < 2:
        if _should_log("INFO"):
            print("INFO: Fewer than 2 claims, skipping consolidation", file=sys.stderr)
        _cb({"status": "skipped", "reason": "too_few_claims"})
        return claims, []

    _cb({
        "status": "analyzing",
        "total_claims": len(claims),
    })

    user_prompt = _build_user_prompt(claims, transcript_json)

    if _should_log("INFO"):
        print(f"INFO: Consolidating {len(claims)} claims via LLM ({model})", file=sys.stderr)

    try:
        raw = ollama_chat(
            ollama_base, model, SYSTEM, user_prompt,
            temperature=temperature,
            force_json=True,
            num_predict=4096,
            show_progress=True,
        )

        if not raw or not raw.strip():
            print("WARNING: Consolidation LLM returned empty response, skipping", file=sys.stderr)
            _cb({"status": "skipped", "reason": "empty_response"})
            return claims, []

        result = extract_json(raw)
        if not isinstance(result, dict):
            print(f"WARNING: Consolidation returned {type(result).__name__}, expected dict. Skipping.", file=sys.stderr)
            _cb({"status": "skipped", "reason": "bad_format"})
            return claims, []

        # Save raw consolidation output for debugging
        debug_dir = "cache/debug"
        os.makedirs(debug_dir, exist_ok=True)
        with open(os.path.join(debug_dir, "consolidate_raw.json"), "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

    except Exception as e:
        print(f"WARNING: Consolidation failed ({type(e).__name__}: {e}). Keeping original claims.", file=sys.stderr)
        _cb({"status": "error", "reason": str(e)})
        return claims, []

    # Apply dedup + grouping
    consolidated, groups = _apply_consolidation(claims, result)

    dupes_removed = len(claims) - len(consolidated)
    if _should_log("INFO"):
        print(f"INFO: Consolidation complete — {dupes_removed} duplicates removed, {len(groups)} narrative groups formed", file=sys.stderr)

    _cb({
        "status": "done",
        "total_claims": len(consolidated),
        "duplicates_removed": dupes_removed,
        "groups": len(groups),
    })

    return consolidated, groups
