# app/pipeline/extract_claims.py
import hashlib
import json
import os
import re
from typing import List
from app.tools.ollama_client import ollama_chat
from app.tools.json_extract import extract_json
from app.policy import FACT_CHECKING_STANDARDS
from app.schemas.claim import Claim


def _should_log(level: str) -> bool:
    """Check if we should log at the given level."""
    levels = {"DEBUG": 0, "INFO": 1, "WARNING": 2, "ERROR": 3}
    current_level = os.environ.get("EVIDENT_LOG_LEVEL", "INFO")
    return levels.get(level, 1) >= levels.get(current_level, 1)


SYSTEM = f"""You are a claim extraction system. Extract ALL checkable factual claims from the provided transcript.

DO NOT extract: opinions, predictions, rhetoric, value judgments
DO extract: statistics, dates, attributions, causal claims, scientific claims, policy claims

CRITICAL EXTRACTION RULES:
1. Extract EVERY checkable claim - do not skip any
2. Make claims COMPLETE and SELF-CONTAINED - include what numbers refer to
3. Use surrounding context to clarify vague claims
4. COMBINE claims that form a single logical argument in the same sentence/context

EXTRACTION WORKFLOW — follow these steps IN ORDER:
1. Read through the transcript and identify exact quotes that contain factual claims
2. For each quote, formulate a complete, self-contained claim
3. The quote_from_transcript field MUST be an EXACT substring copied from the transcript — do not paraphrase or rephrase
4. The claim_text field should expand the quote into a complete, verifiable statement using surrounding context

When to COMBINE claims into ONE:
- Causal relationships: "Without X, then Y" or "X causes Y"
- Conditional statements: "If X then Y"
- Dependent claims: second claim needs first for context
→ Extract as ONE compound claim capturing the full argument

When to keep claims SEPARATE:
- Independent facts stated separately
- Claims that can be verified independently
→ Extract as separate claims

BAD extraction (splitting related claims):
- Claim 1: "There are 90 million illegal migrants"
- Claim 2: "Progressive left gets 30% of vote without them"
→ These form ONE causal argument! Should be combined.

GOOD extraction (combined):
- "The progressive left gets 30% of the vote due to 90 million illegal migrants in the country"
→ Single claim capturing full causal argument (claim_type: causal)

Other examples:
BAD: "He'll bring in 200 million" → INCOMPLETE (what?)
GOOD: "Gavin Newsom will bring in 200 million immigrants" → COMPLETE

BAD: "Crime is up 50%" → INCOMPLETE (where? when?)
GOOD: "Crime is up 50% in Los Angeles in 2024" → COMPLETE

If a claim uses a number or percentage without context, look at surrounding text to complete it.
Extract the COMPLETE version, not fragments.

Output ONLY a JSON array of claim OBJECTS (not strings). No other text.

CRITICAL: Each array element must be an OBJECT with these fields:
- claim_id (string)
- segment_id (string)
- timestamp (null or string)
- claim_text (string) - MUST be complete and self-contained
- quote_from_transcript (string) - MUST be an exact verbatim substring from the transcript
- claim_type (string: statistic|event_date|quote_attribution|causal|medical_science|policy_legal|study_says|biography|other)
- entities (array of strings)
- check_priority (string: high|medium|low)
- needs_context (array of strings)

Example - extract ALL claims in this format:
[
  {{"claim_id": "C001", "segment_id": "S001", "timestamp": null, "claim_text": "specific complete factual claim", "quote_from_transcript": "exact verbatim quote copied from the transcript text", "claim_type": "statistic", "entities": [], "check_priority": "high", "needs_context": []}},
  {{"claim_id": "C002", "segment_id": "S002", "timestamp": null, "claim_text": "another complete factual claim", "quote_from_transcript": "exact verbatim quote copied from the transcript text", "claim_type": "event_date", "entities": [], "check_priority": "medium", "needs_context": []}}
]

{FACT_CHECKING_STANDARDS}

IMPORTANT: Return array of OBJECTS, not strings. Each element must have all required fields.
Extract EVERY claim. Make each claim COMPLETE. Quotes MUST be exact substrings from the transcript."""


# ---------------------------------------------------------------------------
# Normalization & seeding helpers
# ---------------------------------------------------------------------------

def _chunk_seed(chunk_json):
    """Derive a deterministic seed from chunk content."""
    text = json.dumps(chunk_json, sort_keys=True, ensure_ascii=False)
    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)


def _normalize_claim_text(text):
    """Normalize claim text to reduce surface variation between runs."""
    # Standardize whitespace
    text = " ".join(text.split())
    # Standardize number formats: remove commas in numbers (1,000 → 1000)
    text = re.sub(r'(\d),(\d{3})', r'\1\2', text)
    # Standardize percentages: "50 percent" → "50%"
    text = re.sub(r'(\d+)\s*percent', r'\1%', text, flags=re.IGNORECASE)
    # Standardize billions/millions: normalize spacing
    text = re.sub(r'(\d+)\s+(billion|million|trillion)', r'\1 \2', text, flags=re.IGNORECASE)
    # Strip trailing periods (some runs add them, some don't)
    text = text.rstrip('.')
    return text.strip()


# ---------------------------------------------------------------------------
# Chunk extraction
# ---------------------------------------------------------------------------

def _extract_from_chunk(ollama_base: str, model: str, chunk_json: dict,
                        temperature: float, seed: int = None,
                        max_retries: int = 2) -> list:
    """Extract claims from a single chunk of segments."""
    import sys
    import time
    user_prompt = json.dumps(chunk_json, ensure_ascii=False)
    user_with_instruction = (
        f"{user_prompt}\n\n"
        "IMPORTANT: Return a JSON array with ALL checkable factual claims from "
        "these segments. Make each claim COMPLETE and SELF-CONTAINED - if a claim "
        "mentions a number or percentage, include what it refers to by using "
        "surrounding context. COMBINE claims that form a single causal or "
        "conditional argument (like 'Without X then Y') into ONE compound claim. "
        "The quote_from_transcript MUST be copied exactly from the transcript text."
    )

    if _should_log("DEBUG"):
        print(f"DEBUG: Processing chunk with {len(chunk_json.get('segments', []))} segments", file=sys.stderr)

    # Retry logic for empty responses
    for attempt in range(max_retries + 1):
        raw = ollama_chat(
            ollama_base, model, SYSTEM, user_with_instruction,
            temperature=temperature, force_json=False,
            num_predict=8192, show_progress=True, seed=seed,
        )

        # If we got content, break out of retry loop
        if raw and raw.strip():
            break

        # Empty response - retry if we have attempts left
        if attempt < max_retries:
            if _should_log("INFO"):
                print(f"INFO: Empty response on attempt {attempt + 1}, retrying after 3s...", file=sys.stderr)
            time.sleep(3)
        else:
            if _should_log("DEBUG"):
                print(f"DEBUG: All retry attempts exhausted, chunk returned empty", file=sys.stderr)
            return []

    try:
        data = extract_json(raw)
        if not isinstance(data, list):
            if isinstance(data, dict) and any(k in data for k in ["claims", "data", "results", "items"]):
                for key in ["claims", "data", "results", "items"]:
                    if key in data and isinstance(data[key], list):
                        data = data[key]
                        break
            elif isinstance(data, dict) and {"claim_id", "claim_text", "claim_type"}.issubset(data.keys()):
                data = [data]
            else:
                return []

        # Convert string arrays to objects
        if data and isinstance(data[0], str):
            converted = []
            for i, claim_text in enumerate(data, start=1):
                converted.append({
                    "claim_id": f"C{i:03d}",
                    "segment_id": chunk_json["segments"][0]["id"] if chunk_json.get("segments") else "S001",
                    "timestamp": None,
                    "claim_text": claim_text,
                    "quote_from_transcript": claim_text[:200],
                    "claim_type": "other",
                    "entities": [],
                    "check_priority": "medium",
                    "needs_context": []
                })
            data = converted

        # Normalize claim text for consistency
        for item in data:
            if isinstance(item, dict) and "claim_text" in item:
                item["claim_text"] = _normalize_claim_text(item["claim_text"])

        return data if isinstance(data, list) else []
    except Exception as e:
        if _should_log("DEBUG"):
            print(f"DEBUG: Chunk extraction failed: {e}", file=sys.stderr)
        return []


def _deduplicate_claims(claims_data: list) -> list:
    """
    Remove duplicate claims that appeared in multiple overlapping chunks.
    Uses text similarity on normalized text to identify duplicates.
    """
    import difflib
    import sys

    if not claims_data:
        return claims_data

    unique_claims = []
    seen_texts = []

    for claim in claims_data:
        claim_text = _normalize_claim_text(
            claim.get("claim_text", "")
        ).lower()
        if not claim_text:
            continue

        # Check if this claim is too similar to any already seen claim
        is_duplicate = False
        for seen_text in seen_texts:
            similarity = difflib.SequenceMatcher(None, claim_text, seen_text).ratio()
            if similarity > 0.85:  # 85% similarity threshold
                is_duplicate = True
                if _should_log("DEBUG"):
                    print(f"DEBUG: Skipping duplicate claim (similarity={similarity:.2f}): {claim_text[:60]}...", file=sys.stderr)
                break

        if not is_duplicate:
            unique_claims.append(claim)
            seen_texts.append(claim_text)

    return unique_claims


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def extract_claims(ollama_base: str, model: str, transcript_json: dict,
                   max_claims: int = 50, temperature: float = 0.0,
                   chunk_size: int = 20, chunk_overlap: int = 8,
                   progress_callback=None) -> List[Claim]:
    import sys
    import os

    def _cb(data):
        if progress_callback:
            progress_callback(data)

    segments = transcript_json.get("segments", [])

    # If transcript is small, process as single chunk
    if len(segments) <= chunk_size:
        _cb({"chunk": 1, "total_chunks": 1, "status": "extracting", "claims_so_far": 0})

        seed = _chunk_seed(transcript_json)

        # Original single-pass logic
        user_prompt = json.dumps(transcript_json, ensure_ascii=False)
        user_with_instruction = (
            f"{user_prompt}\n\n"
            "IMPORTANT: Return a JSON array with MULTIPLE claims (not just one). "
            "Review ALL segments and extract EVERY checkable factual claim. "
            "Make each claim COMPLETE and SELF-CONTAINED - if a claim mentions a "
            "number or percentage, include what it refers to by using surrounding "
            "context. COMBINE claims that form a single causal or conditional "
            "argument (like 'Without X then Y') into ONE compound claim. "
            "The quote_from_transcript MUST be copied exactly from the transcript text."
        )

        if _should_log("DEBUG"):
            print(f"DEBUG: Sending {len(user_prompt)} chars, {len(segments)} segments to {model}", file=sys.stderr)

        raw = ollama_chat(
            ollama_base, model, SYSTEM, user_with_instruction,
            temperature=temperature, force_json=False,
            num_predict=8192, show_progress=True, seed=seed,
        )

        _cb({"chunk": 1, "total_chunks": 1, "status": "done"})
    else:
        # Chunk-based extraction for larger transcripts with overlapping chunks
        if _should_log("INFO"):
            print(f"INFO: Processing {len(segments)} segments in chunks of {chunk_size} with {chunk_overlap} segment overlap", file=sys.stderr)

        all_claims_data = []
        chunk_step = chunk_size - chunk_overlap  # How far to advance for each chunk
        num_chunks = (len(segments) - chunk_overlap + chunk_step - 1) // chunk_step

        _cb({"chunk": 0, "total_chunks": num_chunks, "status": "starting", "claims_so_far": 0})

        for i in range(num_chunks):
            # Calculate start with overlap from previous chunk
            start_idx = i * chunk_step
            end_idx = min(start_idx + chunk_size, len(segments))

            # Skip if we've already processed all segments
            if start_idx >= len(segments):
                break

            _cb({"chunk": i + 1, "total_chunks": num_chunks, "status": "extracting", "claims_so_far": len(all_claims_data)})

            chunk_segments = segments[start_idx:end_idx]

            chunk_json = {
                "video": transcript_json.get("video", {}),
                "segments": chunk_segments
            }

            seed = _chunk_seed(chunk_json)
            chunk_claims = _extract_from_chunk(ollama_base, model, chunk_json, temperature, seed=seed)
            all_claims_data.extend(chunk_claims)

            _cb({"chunk": i + 1, "total_chunks": num_chunks, "status": "chunk_done", "claims_so_far": len(all_claims_data), "chunk_claims": len(chunk_claims)})

            if _should_log("INFO"):
                print(f"INFO: Chunk {i+1}/{num_chunks} (segments {start_idx}-{end_idx-1}): extracted {len(chunk_claims)} claims", file=sys.stderr)

        # Deduplicate claims from overlapping chunks
        if _should_log("INFO"):
            print(f"INFO: Deduplicating {len(all_claims_data)} claims from overlapping chunks", file=sys.stderr)

        all_claims_data = _deduplicate_claims(all_claims_data)

        if _should_log("INFO"):
            print(f"INFO: After deduplication: {len(all_claims_data)} unique claims", file=sys.stderr)

        # Renumber all claims sequentially to avoid duplicates from different chunks
        for i, claim_data in enumerate(all_claims_data, start=1):
            claim_data["claim_id"] = f"C{i:03d}"

        if _should_log("INFO"):
            print(f"INFO: Renumbered {len(all_claims_data)} claims with unique IDs", file=sys.stderr)

        # Create a fake "raw" response for the rest of the logic
        raw = json.dumps(all_claims_data)

    # Debug: save raw output
    debug_dir = "cache/debug"
    os.makedirs(debug_dir, exist_ok=True)
    with open(os.path.join(debug_dir, "extract_claims_raw.json"), "w", encoding="utf-8") as f:
        f.write(raw)
    if _should_log("DEBUG"):
        print(f"DEBUG: Model returned {len(raw)} chars", file=sys.stderr)

    try:
        data = extract_json(raw)
    except Exception as e:
        print(f"WARNING: Claim extraction JSON parse failed ({type(e).__name__}: {e}). Returning 0 claims.", file=sys.stderr)
        return []

    if not isinstance(data, list):
        print(f"WARNING: Claim extraction returned {type(data).__name__} instead of list.", file=sys.stderr)
        if isinstance(data, dict):
            if _should_log("DEBUG"):
                print(f"DEBUG: Dict keys = {list(data.keys())}", file=sys.stderr)
            # Try to extract array from common wrapper keys
            for key in ["claims", "data", "results", "items"]:
                if key in data and isinstance(data[key], list):
                    if _should_log("DEBUG"):
                        print(f"DEBUG: Found array in data['{key}'], using that instead", file=sys.stderr)
                    data = data[key]
                    break
            else:
                # Check if it looks like a single claim object
                expected_keys = {"claim_id", "claim_text", "claim_type"}
                if expected_keys.issubset(data.keys()):
                    if _should_log("DEBUG"):
                        print(f"DEBUG: Looks like a single claim object, wrapping in array", file=sys.stderr)
                    data = [data]
                else:
                    return []
        else:
            return []

    # Check if array contains strings instead of objects - convert them
    if data and isinstance(data[0], str):
        if _should_log("INFO"):
            print(f"INFO: Model returned {len(data)} string claims. Converting to objects...", file=sys.stderr)
        converted = []
        for i, claim_text in enumerate(data[:max_claims], start=1):
            # Create a minimal claim object from the string
            converted.append({
                "claim_id": f"C{i:03d}",
                "segment_id": "S001",  # Default segment
                "timestamp": None,
                "claim_text": claim_text,
                "quote_from_transcript": claim_text[:200],  # Use first 200 chars as quote
                "claim_type": "other",
                "entities": [],
                "check_priority": "medium",
                "needs_context": []
            })
        data = converted
        if _should_log("INFO"):
            print(f"INFO: Converted {len(data)} string claims to objects", file=sys.stderr)

    claims = []
    valid_claim_types = {"statistic", "event_date", "quote_attribution", "causal", "medical_science", "policy_legal", "study_says", "biography", "other"}
    for i, item in enumerate(data[:max_claims], start=1):
        if not isinstance(item, dict):
            print(f"WARNING: Item {i} is {type(item).__name__}, not dict. Skipping.", file=sys.stderr)
            continue
        item["claim_id"] = item.get("claim_id") or f"C{i:03d}"

        # Normalize claim text for consistency
        if "claim_text" in item:
            item["claim_text"] = _normalize_claim_text(item["claim_text"])

        # Normalize invalid claim_type to 'other'
        if item.get("claim_type") not in valid_claim_types:
            if _should_log("INFO"):
                print(f"INFO: Invalid claim_type '{item.get('claim_type')}' in claim {i}, using 'other'", file=sys.stderr)
            item["claim_type"] = "other"

        try:
            claims.append(Claim(**item))
        except Exception as e:
            print(f"WARNING: Failed to create Claim from item {i}: {e}", file=sys.stderr)
            continue
    return claims
