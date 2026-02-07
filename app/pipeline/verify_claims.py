# app/pipeline/verify_claims.py
import json
import os
from app.tools.ollama_client import ollama_chat
from app.tools.json_extract import extract_json
from app.schemas.verdict import Verdict

SYSTEM = """
Return ONLY valid JSON.
No prose. No markdown. No code fences.

Return exactly ONE JSON object with these keys:
claim_id, rating, confidence, explanation, corrected_claim, severity,
source_tiers_used, red_flags, citations, missing_info, rhetorical_issues

rating MUST be exactly one of:
TRUE, LIKELY TRUE, INSUFFICIENT EVIDENCE, CONFLICTING EVIDENCE, LIKELY FALSE, FALSE

Use INSUFFICIENT EVIDENCE when no relevant or credible evidence was found.
Use CONFLICTING EVIDENCE when multiple credible sources contradict each other on the claim.

CRITICAL - Array fields must be arrays, NOT strings:
- source_tiers_used: [6] or [2, 3] (array of integers)
- red_flags: ["cherry_picked", "outdated"] (array of strings)
- missing_info: ["Need primary source", "Missing date"] (array of strings)
- rhetorical_issues: ["false_causation"] (array of strings)
- citations: array of objects with keys: source_id, snippet_id, tier, url, quote

Example response format (insufficient evidence):
{
  "claim_id": "C001",
  "rating": "INSUFFICIENT EVIDENCE",
  "confidence": 0.3,
  "explanation": "No credible sources were found to verify or refute this claim.",
  "corrected_claim": null,
  "severity": "medium",
  "source_tiers_used": [6],
  "red_flags": ["insufficient_evidence"],
  "citations": [],
  "missing_info": ["Need primary source from official data"],
  "rhetorical_issues": []
}

Example response format (conflicting evidence):
{
  "claim_id": "C002",
  "rating": "CONFLICTING EVIDENCE",
  "confidence": 0.4,
  "explanation": "Multiple credible sources disagree on this claim. Source A reports X while Source B reports Y.",
  "corrected_claim": null,
  "severity": "medium",
  "source_tiers_used": [3, 5],
  "red_flags": ["conflicting_sources"],
  "citations": [
    {
      "source_id": "SRC0001",
      "snippet_id": "SNIP00001",
      "tier": 3,
      "url": "https://example.gov/report",
      "quote": "Quote supporting the claim..."
    },
    {
      "source_id": "SRC0002",
      "snippet_id": "SNIP00005",
      "tier": 5,
      "url": "https://example.com/article",
      "quote": "Quote contradicting the claim..."
    }
  ],
  "missing_info": [],
  "rhetorical_issues": []
}

CRITICAL CITATION RULES:
1. If you use ANY evidence snippet in your reasoning, you MUST cite it in citations array
2. When rating FALSE/LIKELY FALSE/TRUE/LIKELY TRUE, you MUST include citations
3. Extract snippet_id, source_id, tier, url from the evidence_snippets provided
4. Include a relevant quote (excerpt from the snippet) in each citation
5. Do not invent source_id or snippet_id; use only those provided in evidence_snippets
6. Only use INSUFFICIENT EVIDENCE with empty citations if evidence snippets are genuinely irrelevant
7. Use CONFLICTING EVIDENCE (with citations from both sides) when credible sources disagree

Example with citations:
{
  "claim_id": "C003",
  "rating": "FALSE",
  "confidence": 0.9,
  "explanation": "Evidence shows this claim is false...",
  "citations": [
    {
      "source_id": "SRC0007",
      "snippet_id": "SNIP00027",
      "tier": 6,
      "url": "https://example.com/article",
      "quote": "Relevant excerpt from the source that contradicts the claim..."
    }
  ]
}

Rhetorical manipulation detection:
- Check if claim is used to support false causation
- Detect cherry-picked statistics (missing contrary data)
- Identify correlation presented as causation
- Flag misleading framing even if individual fact is true
- Note if surrounding context changes meaning

If claim is TRUE but used misleadingly:
- rating: can be TRUE but add rhetorical_issues
- rhetorical_issues: ["false_causation", "cherry_picked"] (array format)
- explanation: note both truth of claim AND misuse

Examples of rhetorical issues to detect:
- "false_causation": true fact used to imply false cause
- "cherry_picked": selective data omitting contradictory evidence
- "missing_context": true but misleading without full picture
- "correlation_as_causation": confusing correlation with cause
- "appeal_to_fear": using fear to bypass critical thinking
- "false_dichotomy": presenting false either/or choice

Rules:
- Use ONLY the evidence snippets provided
- Check transcript_context to see how claim is being used rhetorically
- Be confident in FALSE ratings when evidence clearly contradicts the claim
- For obviously false claims (e.g., "90 million" when data shows ~11 million), use FALSE not INSUFFICIENT EVIDENCE
- If no relevant evidence found, use INSUFFICIENT EVIDENCE
- If credible sources contradict each other, use CONFLICTING EVIDENCE
- ALWAYS cite evidence snippets that inform your rating (use citations array)
"""

RETRY_SYSTEM = """
Return ONLY valid JSON following the EXACT schema below.
Your response must be a single JSON object and must start with '{' and end with '}'.
No other text. No prose. No markdown. No code fences.

Return exactly ONE JSON object with these EXACT keys (do not change key names):
claim_id, rating, confidence, explanation, corrected_claim, severity,
source_tiers_used, red_flags, citations, missing_info, rhetorical_issues

Required field types:
- claim_id: string
- rating: MUST be exactly one of: TRUE, LIKELY TRUE, INSUFFICIENT EVIDENCE, CONFLICTING EVIDENCE, LIKELY FALSE, FALSE
- confidence: number between 0 and 1
- explanation: string (required, must not be empty)
- corrected_claim: string or null
- severity: MUST be exactly one of: high, medium, low
- source_tiers_used: array of integers
- red_flags: array of strings
- citations: array of objects with keys: source_id, snippet_id, tier, url, quote
- missing_info: array of strings
- rhetorical_issues: array of strings

CRITICAL: Use these EXACT key names. Do NOT use alternatives like:
- "verification_result" (use "rating")
- "verification_reason" (use "explanation")
- Any other variations

If citations is empty, rating MUST be INSUFFICIENT EVIDENCE and confidence <= 0.4.
"""

def _compact_snippets(snips, max_snips: int = 6, max_chars: int = 500):
    snips = sorted(snips or [], key=lambda s: s.get("relevance_score", 0), reverse=True)[:max_snips]
    compact = []
    for s in snips:
        compact.append({
            "snippet_id": s.get("snippet_id"),
            "source_id": s.get("source_id"),
            "tier": s.get("tier"),
            "url": s.get("url"),
            "relevance_score": s.get("relevance_score"),
            "excerpt": (s.get("excerpt") or "")[:max_chars],
        })
    return compact

def _coerce_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        v = value.strip()
        if not v:
            return []
        if "," in v:
            return [x.strip() for x in v.split(",") if x.strip()]
        return [v]
    return [str(value)]

def _coerce_int_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        out = []
        for v in value:
            try:
                out.append(int(v))
            except Exception:
                pass
        return out
    if isinstance(value, str):
        parts = [p.strip() for p in value.split(",") if p.strip()]
        out = []
        for p in parts:
            try:
                out.append(int(p))
            except Exception:
                pass
        return out
    try:
        return [int(value)]
    except Exception:
        return []

def _normalize(data: dict, claim_id: str, sent_snippets: list) -> dict:
    data = data or {}
    data.setdefault("claim_id", claim_id)

    # Explanation mapping
    if "explanation" not in data or not isinstance(data.get("explanation"), str):
        for k in ["analysis", "reason", "rationale", "notes", "summary"]:
            if isinstance(data.get(k), str) and data[k].strip():
                data["explanation"] = data[k].strip()
                break
    if "explanation" not in data:
        data["explanation"] = "Model did not provide an explanation."

    # Rating normalization / inference
    rating = data.get("rating")
    if isinstance(rating, str):
        rating = rating.upper().replace(".", "").strip()
    else:
        rating = None

    allowed = {"TRUE", "LIKELY TRUE", "INSUFFICIENT EVIDENCE", "CONFLICTING EVIDENCE", "LIKELY FALSE", "FALSE"}
    if rating not in allowed:
        verdict_hint = None
        for k in ["verdict", "label", "result", "classification"]:
            if isinstance(data.get(k), str):
                verdict_hint = data[k].upper().strip()
                break
        if verdict_hint in allowed:
            rating = verdict_hint
        elif verdict_hint == "VERIFIED":
            rating = "TRUE"
        elif verdict_hint == "FALSE":
            rating = "FALSE"
        else:
            rating = "INSUFFICIENT EVIDENCE"
    data["rating"] = rating

    # Confidence
    conf = data.get("confidence")
    try:
        conf = float(conf)
    except Exception:
        conf = 0.4 if data["rating"] in ("INSUFFICIENT EVIDENCE", "CONFLICTING EVIDENCE") else 0.6
    conf = max(0.0, min(1.0, conf))
    data["confidence"] = conf

    # Defaults
    data.setdefault("corrected_claim", None)
    sev = data.get("severity")
    if not isinstance(sev, str) or sev.lower() not in ["high", "medium", "low"]:
        data["severity"] = "medium"
    else:
        data["severity"] = sev.lower()

    # Coerce list-ish fields
    data["source_tiers_used"] = _coerce_int_list(data.get("source_tiers_used"))
    data["red_flags"] = _coerce_list(data.get("red_flags"))
    data["missing_info"] = _coerce_list(data.get("missing_info"))
    data["rhetorical_issues"] = _coerce_list(data.get("rhetorical_issues"))

    # citations default then coercion
    data.setdefault("citations", [])
    citations = data.get("citations", [])

    # Build lookup tables from snippets we sent
    snip_by_id = {}
    snips_by_source = {}
    for s in (sent_snippets or []):
        sid = s.get("snippet_id")
        if sid:
            snip_by_id[sid] = s
        src = s.get("source_id")
        if src:
            snips_by_source.setdefault(src, []).append(s)

    fixed_citations = []
    if isinstance(citations, list):
        for c in citations:
            if isinstance(c, dict):
                c.setdefault("source_id", c.get("sourceId") or "")
                c.setdefault("snippet_id", c.get("snippetId") or "")
                c.setdefault("tier", c.get("tier") or 6)
                c.setdefault("url", c.get("url") or "")
                c.setdefault("quote", c.get("quote") or "")
                try:
                    c["tier"] = int(c.get("tier", 6))
                except Exception:
                    c["tier"] = 6
                fixed_citations.append({
                    "source_id": c["source_id"],
                    "snippet_id": c["snippet_id"],
                    "tier": c["tier"],
                    "url": c["url"],
                    "quote": c["quote"],
                })
            elif isinstance(c, str):
                token = c.strip()
                # snippet id
                if token in snip_by_id:
                    s = snip_by_id[token]
                    fixed_citations.append({
                        "source_id": s.get("source_id"),
                        "snippet_id": s.get("snippet_id"),
                        "tier": int(s.get("tier") or 6),
                        "url": s.get("url"),
                        "quote": (s.get("excerpt") or "")[:240],
                    })
                # source id
                elif token in snips_by_source:
                    s = sorted(snips_by_source[token], key=lambda x: x.get("relevance_score", 0), reverse=True)[0]
                    fixed_citations.append({
                        "source_id": s.get("source_id"),
                        "snippet_id": s.get("snippet_id"),
                        "tier": int(s.get("tier") or 6),
                        "url": s.get("url"),
                        "quote": (s.get("excerpt") or "")[:240],
                    })
                else:
                    pass
    else:
        fixed_citations = []

    data["citations"] = fixed_citations

    # Enforce evidence gate (baseline): no citations => INSUFFICIENT EVIDENCE low confidence
    # EXCEPTION: Allow FALSE/LIKELY FALSE ratings when evidence clearly contradicts claim
    if not data.get("citations"):
        current_rating = data.get("rating", "INSUFFICIENT EVIDENCE")
        # Allow negative ratings (FALSE, LIKELY FALSE) to stand even without citations
        # if the model determined the claim is contradicted by evidence
        if current_rating not in ["FALSE", "LIKELY FALSE"]:
            data["rating"] = "INSUFFICIENT EVIDENCE"
            data["confidence"] = min(data["confidence"], 0.4)
            if not data["missing_info"]:
                data["missing_info"] = ["Need at least one reputable source snippet relevant to the claim."]
        else:
            # Keep FALSE/LIKELY FALSE but lower confidence slightly
            data["confidence"] = min(data["confidence"], 0.75)
            # Note: evidence was found but not properly cited
            if "Evidence found but citations incomplete" not in data.get("red_flags", []):
                data["red_flags"].append("evidence_found_citations_incomplete")

    return data

def _norm_claim_type(claim_type: str) -> str:
    ct = (claim_type or "").strip().lower()
    if not ct:
        return "other"
    # normalize common variants
    if "med" in ct or "health" in ct or "clinical" in ct:
        return "medical"
    if "stat" in ct or "number" in ct or "quant" in ct or "economic" in ct or "economy" in ct:
        return "statistical"
    if "caus" in ct or "cause" in ct or "lead to" in ct:
        return "causal"
    if "hist" in ct or "date" in ct or "timeline" in ct:
        return "historical"
    if "defin" in ct or "meaning" in ct:
        return "definition"
    return ct

def _apply_archetype_gate(data: dict, claim_type: str) -> dict:
    """
    Hard rules that enforce evidence quality based on claim archetype.
    ASYMMETRIC: Strict tier requirements for positive ratings (TRUE, LIKELY TRUE),
    relaxed for negative ratings (FALSE, LIKELY FALSE) since any reliable evidence can disprove.
    """
    ct = _norm_claim_type(claim_type)
    current_rating = data.get("rating", "INSUFFICIENT EVIDENCE")

    # Only apply strict tier gates to positive ratings
    # Negative ratings (FALSE, LIKELY FALSE) can use any tier if evidence clearly contradicts
    is_positive_rating = current_rating in ["TRUE", "LIKELY TRUE"]

    cits = data.get("citations") or []
    tiers = []
    for c in cits:
        try:
            tiers.append(int(c.get("tier", 6)))
        except Exception:
            tiers.append(6)

    # helpers
    def has_tier_at_most(n: int) -> bool:
        return any(t <= n for t in tiers)

    def at_least_n_citations(n: int) -> bool:
        return len(cits) >= n

    gate_failed = False
    missing = []

    # Tier meaning in your system:
    # 1 = scholarly journals, 2 = .edu, 3 = gov/WHO/OECD/WB, 4 = research orgs, 5 = news, 6 = everything else
    if ct == "medical":
        # Require at least one citation for all ratings
        if not at_least_n_citations(1):
            gate_failed = True
            missing.append("Medical/health claims require at least one citation.")
        # Only require tier <= 3 for positive ratings
        elif is_positive_rating and not has_tier_at_most(3):
            gate_failed = True
            missing.append("Medical/health claims require a higher-quality source (tier 1–3 such as journals, .edu, .gov, WHO) for verification.")

    elif ct == "statistical":
        # Require at least one citation for all ratings
        if not at_least_n_citations(1):
            gate_failed = True
            missing.append("Statistical claims require at least one citation.")
        # Only require tier <= 3 for positive ratings
        elif is_positive_rating and not has_tier_at_most(3):
            gate_failed = True
            missing.append("Statistical claims require an authoritative source (tier 1–3) for verification, not general websites.")

    elif ct == "causal":
        # Require at least 2 citations for positive ratings, 1 for negative
        min_citations = 2 if is_positive_rating else 1
        if not at_least_n_citations(min_citations):
            gate_failed = True
            if is_positive_rating:
                missing.append("Causal claims require at least two independent citations for verification.")
            else:
                missing.append("Causal claims require at least one citation to disprove.")
        # Only require tier <= 3 for positive ratings
        if cits and is_positive_rating and not has_tier_at_most(3):
            gate_failed = True
            missing.append("Causal claims require at least one higher-quality source (tier 1–3) for verification.")

    elif ct == "historical":
        # Keep baseline rule only (>=1 citation), allow any tier
        if not at_least_n_citations(1):
            gate_failed = True
            missing.append("Historical claims require at least one citation.")

    # definition/other: baseline already enforced.

    if gate_failed:
        data["rating"] = "INSUFFICIENT EVIDENCE"
        data["confidence"] = min(float(data.get("confidence", 0.4)), 0.4)
        # make sure missing_info is a list
        mi = data.get("missing_info")
        if not isinstance(mi, list):
            mi = _coerce_list(mi)
        # add gate reasons (dedup lightly)
        for m in missing:
            if m not in mi:
                mi.append(m)
        data["missing_info"] = mi
        # red flag for traceability
        rf = data.get("red_flags")
        if not isinstance(rf, list):
            rf = _coerce_list(rf)
        if "insufficient_evidence_for_claim_type" not in rf:
            rf.append("insufficient_evidence_for_claim_type")
        data["red_flags"] = rf

    return data

def verify_one(ollama_base: str, model: str, claim, evidence_bundle, outdir: str, temperature: float = 0.0, transcript_json: dict = None) -> Verdict:
    snips = evidence_bundle.get("snippets", [])

    # Fast-fail if no evidence
    if not snips:
        return Verdict(
            claim_id=claim.claim_id,
            rating="INSUFFICIENT EVIDENCE",
            confidence=0.2,
            explanation="No evidence snippets were retrieved for this claim, so it cannot be verified.",
            corrected_claim=None,
            severity="medium",
            source_tiers_used=[],
            red_flags=["anonymous_or_uncited"],
            citations=[],
            missing_info=["Need at least one reputable source snippet relevant to the claim."],
            rhetorical_issues=[]
        )

    compact = _compact_snippets(snips, max_snips=6, max_chars=500)

    # Extract surrounding context from transcript
    transcript_context = ""
    if transcript_json:
        segments = transcript_json.get("segments", [])
        # Find the segment this claim is from
        segment_id = claim.segment_id
        segment_idx = None
        for i, seg in enumerate(segments):
            if seg.get("id") == segment_id:
                segment_idx = i
                break

        if segment_idx is not None:
            # Get 2 segments before and 2 after for context
            context_segments = []
            for i in range(max(0, segment_idx - 2), min(len(segments), segment_idx + 3)):
                seg = segments[i]
                context_segments.append(f"[{seg.get('timestamp', 'N/A')}] {seg.get('speaker', 'Speaker')}: {seg.get('text', '')}")
            transcript_context = "\n".join(context_segments)

    payload = {
        "claim_id": claim.claim_id,
        "claim_text": claim.claim_text,
        "quote_from_transcript": claim.quote_from_transcript,
        "claim_type": claim.claim_type,
        "evidence_snippets": compact,
        "transcript_context": transcript_context,
    }
    user = json.dumps(payload, ensure_ascii=False)

    raw = ollama_chat(
        ollama_base, model, SYSTEM, user,
        temperature=temperature,
        force_json=True,
        timeout_sec=900,
        show_progress=True
    )

    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, f"raw_verifier_{claim.claim_id}.txt"), "w", encoding="utf-8") as f:
        f.write(raw)

    try:
        data = extract_json(raw)
    except Exception:
        try:
            raw2 = ollama_chat(
                ollama_base, model, RETRY_SYSTEM, user,
                temperature=0.0,
                force_json=True,
                timeout_sec=900,
                show_progress=True
            )
            with open(os.path.join(outdir, f"raw_verifier_{claim.claim_id}_retry.txt"), "w", encoding="utf-8") as f:
                f.write(raw2)
            data = extract_json(raw2)
        except Exception as e:
            # Fallback: return INSUFFICIENT EVIDENCE instead of crashing the whole run
            return Verdict(
                claim_id=claim.claim_id,
                rating="INSUFFICIENT EVIDENCE",
                confidence=0.3,
                explanation=f"Verification failed after retry: {type(e).__name__}: {e}",
                corrected_claim=None,
                severity="medium",
                source_tiers_used=[],
                red_flags=["verification_parse_failure"],
                citations=[],
                missing_info=["LLM output could not be parsed into a valid verdict."],
                rhetorical_issues=[]
            )

    data = _normalize(data, claim.claim_id, compact)
    data = _apply_archetype_gate(data, getattr(claim, "claim_type", "") or "")

    return Verdict(**data)


# ---------------------------------------------------------------------------
# Group (narrative) verification
# ---------------------------------------------------------------------------

GROUP_VERIFY_SYSTEM = """
Return ONLY valid JSON. No prose. No markdown. No code fences.

You are a narrative analysis system. You will receive a group of related claims
that a speaker uses together to build a larger argument or narrative.

Your task is to evaluate THE NARRATIVE AS A WHOLE — not the individual claims
(those have already been verified separately).

Ask yourself:
1. Does the narrative thesis logically follow from the individual claims and evidence?
2. Are individually-true claims being assembled to create a MISLEADING narrative?
3. What reasoning gaps exist between the individual facts and the narrative conclusion?
4. Is the rhetorical strategy manipulative even if individual facts are accurate?

Return exactly ONE JSON object with these keys:
- group_id: string (use the group_id provided)
- narrative_thesis: string (restate the narrative thesis)
- narrative_rating: MUST be exactly one of:
  SUPPORTED, PARTIALLY SUPPORTED, MISLEADING, LARGELY MISLEADING, UNSUPPORTED
- narrative_confidence: number between 0 and 1
- explanation: string — explain how the individual claims combine into the narrative
  and whether the narrative conclusion is justified by the evidence
- rhetorical_issues: array of strings — specific manipulation techniques detected
- reasoning_gap: string or null — describe the logical gap between the facts presented
  and the narrative conclusion (null if no gap)
- claim_ids: array of strings — the claim IDs in this group
- individual_ratings_summary: object mapping claim_id to its individual rating

Rating scale:
- SUPPORTED: The narrative conclusion logically follows from the evidence
- PARTIALLY SUPPORTED: Some basis in evidence but overstated or oversimplified
- MISLEADING: Individually-true claims assembled to imply a false or unsupported conclusion
- LARGELY MISLEADING: Multiple false claims combined with rhetorical manipulation
- UNSUPPORTED: The narrative has no evidentiary basis

Example:
{
  "group_id": "G001",
  "narrative_thesis": "Immigration is causing massive fiscal drain on the economy",
  "narrative_rating": "MISLEADING",
  "narrative_confidence": 0.8,
  "explanation": "While C015 correctly cites a real study, the $68,000 figure is disputed. The speaker extrapolates from a single partisan study to claim trillions in losses, ignoring contradictory economic analyses that show net positive fiscal impact of immigration.",
  "rhetorical_issues": ["cherry_picked", "appeal_to_authority", "false_extrapolation"],
  "reasoning_gap": "A single study from a partisan think tank is presented as settled science, ignoring the broader economic consensus and methodological criticisms of the cited study.",
  "claim_ids": ["C015", "C016", "C018"],
  "individual_ratings_summary": {"C015": "LIKELY TRUE", "C016": "LIKELY FALSE", "C018": "MISLEADING"}
}
"""


def verify_group(
    ollama_base: str,
    model: str,
    group,
    claims: list,
    verdicts: list,
    evidence_by_claim: dict,
    transcript_json: dict = None,
    temperature: float = 0.0,
):
    """
    Verify a narrative claim group by evaluating how individual claims
    combine to form a larger argument.

    Args:
        group: ClaimGroup instance
        claims: list of all Claim objects
        verdicts: list of all Verdict objects
        evidence_by_claim: dict mapping claim_id → evidence bundle
        transcript_json: full transcript for context
        temperature: LLM temperature

    Returns:
        GroupVerdict instance
    """
    from app.schemas.verdict import GroupVerdict

    # Build lookup tables
    claim_by_id = {c.claim_id: c for c in claims}
    verdict_by_id = {v.claim_id: v for v in verdicts}

    # Collect member claims and their verdicts
    member_claims = []
    individual_summary = {}
    for cid in group.claim_ids:
        c = claim_by_id.get(cid)
        v = verdict_by_id.get(cid)
        if c:
            entry = {
                "claim_id": cid,
                "claim_text": c.claim_text,
                "quote_from_transcript": c.quote_from_transcript,
                "claim_type": c.claim_type,
            }
            if v:
                entry["rating"] = v.rating
                entry["confidence"] = v.confidence
                entry["explanation"] = v.explanation[:300]
                entry["rhetorical_issues"] = v.rhetorical_issues
                individual_summary[cid] = v.rating
            else:
                entry["rating"] = "NOT VERIFIED"
                individual_summary[cid] = "NOT VERIFIED"
            member_claims.append(entry)

    # Pool evidence snippets from all group members (deduplicated)
    seen_snippet_ids = set()
    pooled_snippets = []
    for cid in group.claim_ids:
        bundle = evidence_by_claim.get(cid, {})
        for snip in bundle.get("snippets", []):
            sid = snip.get("snippet_id")
            if sid and sid not in seen_snippet_ids:
                seen_snippet_ids.add(sid)
                pooled_snippets.append(snip)

    # Take top ~10 by relevance
    pooled_snippets = sorted(
        pooled_snippets,
        key=lambda s: s.get("relevance_score", 0),
        reverse=True
    )[:10]
    compact_evidence = _compact_snippets(pooled_snippets, max_snips=10, max_chars=400)

    # Gather transcript context around all group claims
    transcript_context = ""
    if transcript_json:
        segments = transcript_json.get("segments", [])
        seg_by_id = {s.get("id"): i for i, s in enumerate(segments)}
        context_indices = set()
        for cid in group.claim_ids:
            c = claim_by_id.get(cid)
            if c and c.segment_id in seg_by_id:
                idx = seg_by_id[c.segment_id]
                for i in range(max(0, idx - 1), min(len(segments), idx + 2)):
                    context_indices.add(i)
        if context_indices:
            context_lines = []
            for i in sorted(context_indices):
                seg = segments[i]
                context_lines.append(
                    f"[{seg.get('timestamp', 'N/A')}] {seg.get('speaker', 'Speaker')}: {seg.get('text', '')}"
                )
            transcript_context = "\n".join(context_lines)

    payload = {
        "group_id": group.group_id,
        "narrative_thesis": group.narrative_thesis,
        "rhetorical_strategy": group.rhetorical_strategy,
        "member_claims": member_claims,
        "pooled_evidence": compact_evidence,
        "transcript_context": transcript_context,
    }
    user = json.dumps(payload, ensure_ascii=False)

    raw = ollama_chat(
        ollama_base, model, GROUP_VERIFY_SYSTEM, user,
        temperature=temperature,
        force_json=True,
        timeout_sec=900,
        show_progress=True,
    )

    try:
        data = extract_json(raw)
    except Exception as e:
        # Fallback: return a cautious group verdict
        return GroupVerdict(
            group_id=group.group_id,
            narrative_thesis=group.narrative_thesis,
            narrative_rating="PARTIALLY SUPPORTED",
            narrative_confidence=0.3,
            explanation=f"Group verification failed to parse: {type(e).__name__}: {e}",
            rhetorical_issues=[],
            reasoning_gap=None,
            claim_ids=list(group.claim_ids),
            individual_ratings_summary=individual_summary,
        )

    # Normalize the group verdict data
    data = data or {}
    data["group_id"] = group.group_id
    data["narrative_thesis"] = group.narrative_thesis
    data["claim_ids"] = list(group.claim_ids)
    data["individual_ratings_summary"] = individual_summary

    # Validate narrative_rating
    allowed_ratings = {
        "SUPPORTED", "PARTIALLY SUPPORTED", "MISLEADING",
        "LARGELY MISLEADING", "UNSUPPORTED"
    }
    nr = (data.get("narrative_rating") or "").upper().strip()
    if nr not in allowed_ratings:
        nr = "PARTIALLY SUPPORTED"
    data["narrative_rating"] = nr

    # Validate confidence
    try:
        data["narrative_confidence"] = max(0.0, min(1.0, float(data.get("narrative_confidence", 0.5))))
    except Exception:
        data["narrative_confidence"] = 0.5

    data.setdefault("explanation", "No explanation provided.")
    data.setdefault("rhetorical_issues", [])
    data.setdefault("reasoning_gap", None)

    if isinstance(data.get("rhetorical_issues"), str):
        data["rhetorical_issues"] = [data["rhetorical_issues"]]

    return GroupVerdict(**data)