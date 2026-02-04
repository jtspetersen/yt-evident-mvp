# app/tools/searx.py
import requests
from urllib.parse import urlparse

def _host(url: str) -> str:
    try:
        return (urlparse(url).hostname or "").lower()
    except Exception:
        return ""

def searx_search(base_url: str, query: str, num_results: int = 8, deny_domains=None):
    deny_domains = deny_domains or []
    deny_domains = [d.lower() for d in deny_domains]

    params = {"q": query, "format": "json"}
    r = requests.get(f"{base_url}/search", params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    results = data.get("results", [])

    out = []
    for item in results:
        url = item.get("url")
        if not url or not (url.startswith("http://") or url.startswith("https://")):
            continue

        host = _host(url)
        if host and any(d in host for d in deny_domains):
            continue

        out.append({
            "url": url,
            "title": item.get("title"),
            "score": item.get("score", None),
            "content": item.get("content", None),
        })
        if len(out) >= num_results:
            break

    return out
