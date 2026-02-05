# app/pipeline/retrieve_evidence.py
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from dateutil import tz
from urllib.parse import urlparse
from tqdm import tqdm

from app.tools.searx import searx_search
from app.tools.fetch import fetch_url
from app.tools.parse import extract_text_from_html
from app.tools.snippets import make_snippets, top_k_snippets

# ---------------------------
# Helpers
# ---------------------------

def source_tier_guess(url: str) -> int:
    """
    Assign quality tier to source URL. Lower tier = higher quality.

    Tier 1: Top scholarly journals and academic publishers
    Tier 2: Academic institutions and university research
    Tier 3: Government and international organizations
    Tier 4: Major research organizations and think tanks
    Tier 5: Established news agencies and fact-checkers
    Tier 6: Everything else
    """
    u = (url or "").lower()

    # Tier 1: Top scholarly journals and academic publishers
    tier1_domains = [
        "nature.com", "science.org", "sciencemag.org", "cell.com",
        "thelancet.com", "nejm.org", "bmj.com", "jama.jamanetwork.com",
        "springer.com", "sciencedirect.com", "wiley.com",
        "oup.com", "cambridge.org", "jstor.org", "arxiv.org",
        "plos.org", "frontiersin.org", "mdpi.com"
    ]
    if any(d in u for d in tier1_domains):
        return 1

    # Tier 2: Academic institutions (.edu, .ac.uk, etc.)
    if ".edu" in u or ".ac.uk" in u or ".ac." in u:
        return 2

    # Tier 3: Government and major international organizations
    tier3_domains = [
        ".gov", "who.int", "un.org", "oecd.org", "worldbank.org",
        "imf.org", "europa.eu", "cdc.gov", "nih.gov", "census.gov"
    ]
    if any(d in u for d in tier3_domains):
        return 3

    # Tier 4: Major research organizations, think tanks, polling organizations
    tier4_domains = [
        "pewresearch.org", "pewsocialtrends.org", "pewtrusts.org",
        "brookings.edu", "rand.org", "cfr.org", "carnegieendowment.org",
        "heritage.org", "aei.org", "cato.org", "urban.org",
        "kff.org", "gallup.com", "ipsos.com", "yougov.com",
        "statista.com", "ourworldindata.org", "factcheck.org",
        "politifact.com", "snopes.com", "usafacts.org"
    ]
    if any(d in u for d in tier4_domains):
        return 4

    # Tier 5: Established news agencies and major outlets
    tier5_domains = [
        "reuters.com", "apnews.com", "ap.org", "bbc.com", "bbc.co.uk",
        "economist.com", "ft.com", "wsj.com", "nytimes.com",
        "washingtonpost.com", "theguardian.com", "npr.org", "pbs.org"
    ]
    if any(d in u for d in tier5_domains):
        return 5

    # Tier 6: Everything else
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
        # Forum patterns
        "/forum/", "/forums/", "/thread/", "/topic/", "/discussion/",
        "/showthread", "/viewtopic", "/board/", "/boards/",
        # User-generated content patterns
        "/user/", "/profile/", "/member/", "/members/",
        # Comment sections and social features
        "/comment/", "/comments/", "/reply/", "/replies/",
        # Blog patterns (combined with deny_domains for blog platforms)
        "/author/", "/contributor/",
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

def _fetch_and_parse(url, timeout_sec):
    """Fetch a URL and extract text. Runs in a worker thread."""
    html, status, err = fetch_url(url, timeout_sec=timeout_sec)
    if html is None:
        return {"ok": False, "url": url, "status": status, "error": err, "stage": "fetch"}

    if is_probably_not_html(html):
        return {"ok": False, "url": url, "status": status, "error": "content looks like xml/rss/sitemap", "stage": "parse"}

    text = extract_text_from_html(html)
    if not text or len(text) < 200:
        return {"ok": False, "url": url, "status": status, "error": "extracted text empty/too short", "stage": "extract_text"}

    return {"ok": True, "url": url, "status": status, "text": text}


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
    fetch_timeout = int(budgets["fetch_timeout_sec"])

    deny_domains = [d.lower() for d in (deny_domains or [])]
    query_overrides = query_overrides or {}

    # Track repeated domain failures to avoid burning budget on a single blocked host
    domain_fail_counts = {}   # host -> count
    domain_blocked = set()    # hosts to skip during this run
    max_failures_per_domain = int(budgets.get("max_failures_per_domain", 6))

    fetch_workers = int(budgets.get("fetch_workers", 8))

    claim_bar = tqdm(
        claims,
        desc="Retrieving evidence",
        unit=" claim",
        bar_format="{desc}: {n_fmt}/{total_fmt} claims | {elapsed} | {postfix}",
        leave=False,
    )
    claim_bar.set_postfix_str("src=0 snip=0 fail=0")

    with ThreadPoolExecutor(max_workers=fetch_workers) as executor:
        for c in claim_bar:
            query = query_overrides.get(c.claim_id) or c.claim_text

            results = searx_search(
                searx_base,
                query,
                num_results=int(budgets["max_sources_per_claim"]),
                deny_domains=deny_domains
            )

            # --- Filter candidates and submit parallel fetches ---
            candidates = []  # (url, title, host_str)
            for r in results:
                if fetch_count + len(candidates) >= max_fetches:
                    break

                url = r.get("url")
                if not url:
                    continue

                h = host(url)
                if not h:
                    continue
                if h in domain_blocked:
                    continue
                if any(d in h for d in deny_domains):
                    continue
                if looks_like_binary(url) or looks_like_junk_url(url):
                    continue

                candidates.append((url, r.get("title"), h))

            if not candidates:
                evidence_by_claim[c.claim_id] = {
                    "claim_id": c.claim_id,
                    "query": query,
                    "snippets": []
                }
                continue

            claim_bar.set_description(f"Fetching {len(candidates)} URLs")

            # Submit all fetches for this claim concurrently
            future_to_info = {}
            for url, title, h in candidates:
                fut = executor.submit(_fetch_and_parse, url, fetch_timeout)
                future_to_info[fut] = (url, title, h)

            # Process results as they complete
            claim_snips = []
            for fut in as_completed(future_to_info):
                url, title, h = future_to_info[fut]
                fetch_count += 1

                try:
                    result = fut.result()
                except Exception as e:
                    result = {"ok": False, "url": url, "status": None, "error": str(e), "stage": "fetch"}

                if not result["ok"]:
                    domain_fail_counts[h] = domain_fail_counts.get(h, 0) + 1
                    if domain_fail_counts[h] >= max_failures_per_domain:
                        domain_blocked.add(h)

                    fetch_failures.append({
                        "stage": result.get("stage", "fetch"),
                        "url": url,
                        "host": h,
                        "status": result.get("status"),
                        "error": result.get("error"),
                        "claim_id": c.claim_id,
                        "query": query
                    })
                    claim_bar.set_postfix_str(f"src={len(all_sources)} snip={len(all_snippets)} fail={len(fetch_failures)}")
                    continue

                text = result["text"]
                src_id = f"SRC{len(all_sources)+1:04d}"
                tier = source_tier_guess(url)

                src = {
                    "source_id": src_id,
                    "url": url,
                    "title": title,
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

                claim_bar.set_postfix_str(f"src={len(all_sources)} snip={len(all_snippets)} fail={len(fetch_failures)}")

            evidence_by_claim[c.claim_id] = {
                "claim_id": c.claim_id,
                "query": query,
                "snippets": claim_snips
            }
            claim_bar.set_description("Retrieving evidence")

    claim_bar.close()

    return all_sources, all_snippets, evidence_by_claim, fetch_failures
