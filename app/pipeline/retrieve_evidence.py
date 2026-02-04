# app/pipeline/retrieve_evidence.py
import hashlib
from datetime import datetime
from dateutil import tz
from urllib.parse import urlparse

from app.tools.searx import searx_search
from app.tools.fetch import fetch_url
from app.tools.parse import extract_text_from_html
from app.tools.snippets import make_snippets, top_k_snippets

# ---------------------------
# Helpers
# ---------------------------

def source_tier_guess(url: str) -> int:
    u = (url or "").lower()
    if ".gov" in u or "who.int" in u or "oecd.org" in u or "worldbank.org" in u:
        return 3
    if ".edu" in u:
        return 2
    return 6

def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def host(url: str) -> str:
    try:
        return (urlparse(url).hostname or "").lower()
    except Exception:
        return ""

def looks_like_binary(url: str) -> bool:
    u = (url or "").lower()
    # strip query string for extension checks
    u_no_q = u.split("?", 1)[0]
    return any(u_no_q.endswith(ext) for ext in [
        ".pdf", ".zip", ".rar", ".7z",
        ".mp4", ".mov", ".avi", ".mkv",
        ".mp3", ".wav",
        ".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg",
        ".ppt", ".pptx", ".doc", ".docx", ".xls", ".xlsx"
    ])

def looks_like_junk_url(url: str) -> bool:
    """
    Skips pages that are typically not useful for evidence extraction
    and/or commonly cause blocks.
    """
    u = (url or "").lower()
    bad_fragments = [
        "/login", "/signin", "/sign-in", "/account",
        "/subscribe", "/subscription", "/paywall", "/checkout",
        "/tag/", "/tags/", "/category/", "/categories/",
        "/archive", "/archives/",
        "/amp", "/amp/",
        "/feed", "/rss", "sitemap",
        "utm_", "fbclid=", "gclid=",
    ]
    return any(b in u for b in bad_fragments)

def is_probably_not_html(text: str) -> bool:
    """
    Quick heuristic: if it looks like a feed, sitemap, or super short
    non-content page, we'll treat it as non-useful content.
    """
    if not text:
        return True
    t = text.strip().lower()
    if t.startswith("<?xml") or "<rss" in t[:2000] or "<urlset" in t[:2000]:
        return True
    return False

# ---------------------------
# Main retrieval
# ---------------------------

def retrieve_for_claims(
    searx_base: str,
    claims,
    budgets,
    tz_name: str,
    snippets_per_source: int = 4,
    snippet_max_chars: int = 1200,
    deny_domains=None,
    query_overrides=None,
    extra_fetch_budget: int = 0,
):
    retrieved_at = datetime.now(tz.gettz(tz_name)).isoformat()

    all_sources = []
    all_snippets = []
    evidence_by_claim = {}
    fetch_failures = []

    max_fetches = int(budgets["max_fetches_per_run"]) + int(extra_fetch_budget)
    fetch_count = 0

    deny_domains = [d.lower() for d in (deny_domains or [])]
    query_overrides = query_overrides or {}

    # Track repeated domain failures to avoid burning budget on a single blocked host
    domain_fail_counts = {}   # host -> count
    domain_blocked = set()    # hosts to skip during this run
    max_failures_per_domain = int(budgets.get("max_failures_per_domain", 6))

    for c in claims:
        query = query_overrides.get(c.claim_id) or c.claim_text

        results = searx_search(
            searx_base,
            query,
            num_results=int(budgets["max_sources_per_claim"]),
            deny_domains=deny_domains
        )

        claim_snips = []

        for r in results:
            if fetch_count >= max_fetches:
                break

            url = r.get("url")
            if not url:
                continue

            h = host(url)
            if not h:
                continue

            # per-run domain block (repeated 403/429 etc.)
            if h in domain_blocked:
                continue

            # extra denylist safety (in case searx filtering misses variants)
            if any(d in h for d in deny_domains):
                continue

            # skip non-useful urls early
            if looks_like_binary(url) or looks_like_junk_url(url):
                continue

            html, status, err = fetch_url(url, timeout_sec=int(budgets["fetch_timeout_sec"]))
            fetch_count += 1

            if html is None:
                # track domain failures and possibly block
                domain_fail_counts[h] = domain_fail_counts.get(h, 0) + 1
                if domain_fail_counts[h] >= max_failures_per_domain:
                    domain_blocked.add(h)

                fetch_failures.append({
                    "stage": "fetch",
                    "url": url,
                    "host": h,
                    "status": status,
                    "error": err,
                    "claim_id": c.claim_id,
                    "query": query
                })
                continue

            if is_probably_not_html(html):
                fetch_failures.append({
                    "stage": "parse",
                    "url": url,
                    "host": h,
                    "status": status,
                    "error": "content looks like xml/rss/sitemap",
                    "claim_id": c.claim_id,
                    "query": query
                })
                continue

            text = extract_text_from_html(html)
            if not text or len(text) < 200:
                fetch_failures.append({
                    "stage": "extract_text",
                    "url": url,
                    "host": h,
                    "status": status,
                    "error": "extracted text empty/too short",
                    "claim_id": c.claim_id,
                    "query": query
                })
                continue

            src_id = f"SRC{len(all_sources)+1:04d}"
            tier = source_tier_guess(url)

            src = {
                "source_id": src_id,
                "url": url,
                "title": r.get("title"),
                "publisher": h,
                "retrieved_at": retrieved_at,
                "tier": tier,
                "content_hash": sha256(text),
                "content_text": text[:20000],
            }
            all_sources.append(src)

            snippets = make_snippets(text, max_chars=max(600, int(snippet_max_chars)), overlap=200)
            top = top_k_snippets(c.claim_text, snippets, k=int(snippets_per_source))

            for (score, start, end, chunk) in top:
                snip_id = f"SNIP{len(all_snippets)+1:05d}"
                sn = {
                    "snippet_id": snip_id,
                    "source_id": src_id,
                    "url": url,
                    "tier": tier,
                    "relevance_score": round(score, 3),
                    "location": {"type": "text_range", "start_char": start, "end_char": end},
                    "excerpt": (chunk or "")[:int(snippet_max_chars)],
                }
                all_snippets.append(sn)
                claim_snips.append(sn)

        evidence_by_claim[c.claim_id] = {
            "claim_id": c.claim_id,
            "query": query,
            "snippets": claim_snips
        }

    return all_sources, all_snippets, evidence_by_claim, fetch_failures