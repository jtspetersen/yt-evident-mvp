# app/tools/query_gen.py
"""
LLM-based query generation for evidence retrieval.
Converts natural-language claims into 2-3 search-engine-optimized queries.
"""
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from app.tools.ollama_client import ollama_chat
from app.tools.json_extract import extract_json

QUERY_GEN_SYSTEM = """\
You are a search query specialist for fact-checking. Given a factual claim,
generate search queries that would help find evidence to verify or refute it.

Rules:
1. Generate exactly {n} search queries, each optimized for a web search engine
2. Remove editorialized language, rhetorical framing, and opinions
3. Focus on the specific factual core: names, numbers, dates, institutions
4. Vary query angles: one for the direct fact, one for counter-evidence or the primary source
5. Keep queries short (5-12 words) — no full sentences
6. If the claim mentions a study, statistic, or quote attribution, include the source name

Return ONLY a JSON object:
{{"queries": ["query one", "query two", "query three"]}}

Example:
Claim: "The progressive left gets 30% of the vote due to 90 million illegal migrants"
{{"queries": [
  "US undocumented immigrant population 2024 estimate",
  "immigration impact progressive vote share",
  "90 million illegal immigrants fact check"
]}}"""


def generate_queries_for_claim(
    ollama_base: str,
    model: str,
    claim_text: str,
    claim_type: str = "other",
    entities: list = None,
    num_queries: int = 3,
    temperature: float = 0.3,
) -> list:
    """
    Generate search-optimized queries for a single claim.

    Returns list of query strings, or [claim_text] as fallback
    if LLM generation fails.
    """
    system = QUERY_GEN_SYSTEM.replace("{n}", str(num_queries))
    user = json.dumps({
        "claim_text": claim_text,
        "claim_type": claim_type,
        "entities": entities or [],
    }, ensure_ascii=False)

    try:
        raw = ollama_chat(
            ollama_base, model, system, user,
            temperature=temperature,
            force_json=True,
            num_predict=512,
            timeout_sec=60,
        )
        data = extract_json(raw)
        queries = data.get("queries", [])
        if isinstance(queries, list) and queries:
            seen = set()
            unique = []
            for q in queries:
                if not isinstance(q, str):
                    continue
                q_clean = q.strip()
                q_lower = q_clean.lower()
                if q_lower and q_lower not in seen:
                    seen.add(q_lower)
                    unique.append(q_clean)
            return unique[:num_queries] if unique else [claim_text]
        return [claim_text]
    except Exception:
        return [claim_text]


def generate_queries_batch(
    ollama_base: str,
    model: str,
    claims: list,
    num_queries: int = 3,
    temperature: float = 0.3,
    max_workers: int = 3,
    progress_callback=None,
) -> dict:
    """
    Generate queries for multiple claims in parallel.

    Args:
        claims: list of Claim objects (must have claim_id, claim_text)
        num_queries: queries per claim
        max_workers: parallel LLM calls (bounded by Ollama capacity)
        progress_callback: optional callable({claim_idx, total, claim_id, status, num_queries})

    Returns:
        dict mapping claim_id -> list[str] of queries
    """
    result = {}

    def _gen_one(claim):
        queries = generate_queries_for_claim(
            ollama_base, model,
            claim.claim_text,
            claim_type=getattr(claim, "claim_type", "other"),
            entities=getattr(claim, "entities", []),
            num_queries=num_queries,
            temperature=temperature,
        )
        return claim.claim_id, queries

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_gen_one, c): c for c in claims}
        done_count = 0
        for fut in as_completed(futures):
            try:
                claim_id, queries = fut.result()
            except Exception:
                claim = futures[fut]
                claim_id = claim.claim_id
                queries = [claim.claim_text]

            result[claim_id] = queries
            done_count += 1
            if progress_callback:
                progress_callback({
                    "claim_idx": done_count,
                    "total": len(claims),
                    "claim_id": claim_id,
                    "status": "generated",
                    "num_queries": len(queries),
                })

    return result
