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
source_tiers_used, red_flags, citations, missing_info

rating MUST be exactly one of:
VERIFIED, LIKELY TRUE, UNCERTAIN, LIKELY FALSE, FALSE

citations MUST be a list of objects with keys:
source_id, snippet_id, tier, url, quote

Rules:
- Use ONLY the evidence snippets provided.
- If citations is empty OR missing, rating MUST be UNCERTAIN and confidence <= 0.4.
- Do not invent source_id or snippet_id; use only those provided in evidence_snippets.
"""

RETRY_SYSTEM = """
Return ONLY valid JSON following the required schema and keys.
Your response must be a single JSON object and must start with '{' and end with '}'.
No other text.
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

    allowed = {"VERIFIED", "LIKELY TRUE", "UNCERTAIN", "LIKELY FALSE", "FALSE"}
    if rating not in allowed:
        verdict_hint = None
        for k in ["verdict", "label", "result", "classification"]:
            if isinstance(data.get(k), str):
                verdict_hint = data[k].upper().strip()
                break
        if verdict_hint in allowed:
            rating = verdict_hint
        elif verdict_hint == "TRUE":
            rating = "VERIFIED"
        elif verdict_hint == "FALSE":
            rating = "FALSE"
        else:
            rating = "UNCERTAIN"
    data["rating"] = rating

    # Confidence
    conf = data.get("confidence")
    try:
        conf = float(conf)
    except Exception:
        conf = 0.4 if data["rating"] == "UNCERTAIN" else 0.6
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

    # Enforce evidence gate (baseline): no citations => UNCERTAIN low confidence
    if not data.get("citations"):
        data["rating"] = "UNCERTAIN"
        data["confidence"] = min(data["confidence"], 0.4)
        if not data["missing_info"]:
            data["missing_info"] = ["Need at least one reputable source snippet relevant to the claim."]

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
    This is conservative: failing the gate downgrades to UNCERTAIN.
    """
    ct = _norm_claim_type(claim_type)

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
    # 2 = .edu, 3 = gov/WHO/OECD/WB, 6 = everything else
    if ct == "medical":
        # Require at least one strong source (tier <= 3)
        if not at_least_n_citations(1):
            gate_failed = True
            missing.append("Medical/health claims require at least one citation.")
        elif not has_tier_at_most(3):
            gate_failed = True
            missing.append("Medical/health claims require a higher-quality source (tier 2–3 such as .edu, .gov, WHO, OECD, World Bank).")

    elif ct == "statistical":
        # Require at least one non-junk citation; in your current tiers that basically means <=3
        if not at_least_n_citations(1):
            gate_failed = True
            missing.append("Statistical claims require at least one citation.")
        elif not has_tier_at_most(3):
            gate_failed = True
            missing.append("Statistical claims require an authoritative source (tier 2–3), not general blogs.")

    elif ct == "causal":
        # Stronger: need 2 citations, and at least one strong
        if not at_least_n_citations(2):
            gate_failed = True
            missing.append("Causal claims require at least two independent citations.")
        if cits and not has_tier_at_most(3):
            gate_failed = True
            missing.append("Causal claims require at least one higher-quality source (tier 2–3).")

    elif ct == "historical":
        # Keep baseline rule only (>=1 citation), allow tier 6.
        if not at_least_n_citations(1):
            gate_failed = True
            missing.append("Historical claims require at least one citation.")

    # definition/other: baseline already enforced.

    if gate_failed:
        data["rating"] = "UNCERTAIN"
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

def verify_one(ollama_base: str, model: str, claim, evidence_bundle, outdir: str, temperature: float = 0.0) -> Verdict:
    snips = evidence_bundle.get("snippets", [])

    # Fast-fail if no evidence
    if not snips:
        return Verdict(
            claim_id=claim.claim_id,
            rating="UNCERTAIN",
            confidence=0.2,
            explanation="No evidence snippets were retrieved for this claim, so it cannot be verified.",
            corrected_claim=None,
            severity="medium",
            source_tiers_used=[],
            red_flags=["anonymous_or_uncited"],
            citations=[],
            missing_info=["Need at least one reputable source snippet relevant to the claim."]
        )

    compact = _compact_snippets(snips, max_snips=6, max_chars=500)

    payload = {
        "claim_id": claim.claim_id,
        "claim_text": claim.claim_text,
        "quote_from_transcript": claim.quote_from_transcript,
        "claim_type": claim.claim_type,
        "evidence_snippets": compact,
    }
    user = json.dumps(payload, ensure_ascii=False)

    raw = ollama_chat(
        ollama_base, model, SYSTEM, user,
        temperature=temperature,
        force_json=True,
        timeout_sec=900
    )

    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, f"raw_verifier_{claim.claim_id}.txt"), "w", encoding="utf-8") as f:
        f.write(raw)

    try:
        data = extract_json(raw)
    except Exception:
        raw2 = ollama_chat(
            ollama_base, model, RETRY_SYSTEM, user,
            temperature=0.0,
            force_json=True,
            timeout_sec=900
        )
        with open(os.path.join(outdir, f"raw_verifier_{claim.claim_id}_retry.txt"), "w", encoding="utf-8") as f:
            f.write(raw2)
        data = extract_json(raw2)

    data = _normalize(data, claim.claim_id, compact)
    data = _apply_archetype_gate(data, getattr(claim, "claim_type", "") or "")

    return Verdict(**data)