# app/pipeline/extract_claims.py
import json
from typing import List
from app.tools.ollama_client import ollama_chat
from app.tools.json_extract import extract_json
from app.policy import FACT_CHECKING_STANDARDS
from app.schemas.claim import Claim

SYSTEM = f"""
You are the Claim Extractor for a fact-checking workflow.
Extract only CHECKABLE factual claims from the transcript.
Do NOT include opinions, predictions, vague rhetoric, or value judgments.

Return ONLY valid JSON: a list of objects matching this schema:
{{
  "claim_id": "C001",
  "segment_id": "S001",
  "timestamp": null,
  "claim_text": "...",
  "quote_from_transcript": "...",
  "claim_type": "statistic|event_date|quote_attribution|causal|medical_science|policy_legal|study_says|biography|other",
  "entities": ["..."],
  "check_priority": "high|medium|low",
  "needs_context": ["..."]
}}

Rules:
- Max 50 claims.
- If ambiguous, include needs_context questions.
- Keep quote_from_transcript short and exact.
- claim_text should be a clean paraphrase in plain English.
{FACT_CHECKING_STANDARDS}
"""

def extract_claims(ollama_base: str, model: str, transcript_json: dict, max_claims: int = 50, temperature: float = 0.1) -> List[Claim]:
    user = json.dumps(transcript_json, ensure_ascii=False)
    raw = ollama_chat(ollama_base, model, SYSTEM, user, temperature=temperature)

    try:
        data = extract_json(raw)
    except Exception as e:
        import sys
        print(f"WARNING: Claim extraction JSON parse failed ({type(e).__name__}: {e}). Returning 0 claims.", file=sys.stderr)
        return []

    if not isinstance(data, list):
        import sys
        print(f"WARNING: Claim extraction returned {type(data).__name__} instead of list. Returning 0 claims.", file=sys.stderr)
        return []

    claims = []
    for i, item in enumerate(data[:max_claims], start=1):
        item["claim_id"] = item.get("claim_id") or f"C{i:03d}"
        claims.append(Claim(**item))
    return claims
