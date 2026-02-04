# app/tools/fetch.py
import os
import requests

from app.tools.url_cache import get_cached, set_cached

# Module-level stats you can read after a run
FETCH_STATS = {
    "cache_hit_ok": 0,
    "cache_hit_fail": 0,
    "cache_miss": 0,
    "net_ok": 0,
    "net_fail": 0,
}

def _read_cache_days_from_config(default_days: int = 7) -> int:
    try:
        import yaml
        if os.path.exists("config.yaml"):
            with open("config.yaml", "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            days = (cfg.get("cache", {}) or {}).get("url_cache_days", default_days)
            return int(days)
    except Exception:
        pass
    return int(default_days)

def fetch_url(url: str, timeout_sec: int = 20):
    """
    Returns (html_text | None, status_code | None, error | None)

    Uses disk cache:
      cache/url/<sha>.json
    TTL days from config.yaml cache.url_cache_days (default 7).
    """
    cache_days = _read_cache_days_from_config(7)
    cache_dir = os.path.join("cache", "url")

    cached = get_cached(cache_dir, url, ttl_days=cache_days)
    if cached is not None:
        if cached.get("ok") and cached.get("text"):
            FETCH_STATS["cache_hit_ok"] += 1
            return cached.get("text"), cached.get("status"), None
        FETCH_STATS["cache_hit_fail"] += 1
        return None, cached.get("status"), cached.get("error") or "cached failure"

    FETCH_STATS["cache_miss"] += 1

    headers = {"User-Agent": "Mozilla/5.0 (EvidentBot/0.1; +local)"}

    try:
        r = requests.get(url, headers=headers, timeout=int(timeout_sec))
        status = r.status_code
        if 200 <= status < 300:
            text = r.text
            set_cached(cache_dir, url, status=status, ok=True, text=text, error=None)
            FETCH_STATS["net_ok"] += 1
            return text, status, None
        else:
            err = f"HTTP {status}"
            set_cached(cache_dir, url, status=status, ok=False, text=None, error=err)
            FETCH_STATS["net_fail"] += 1
            return None, status, err
    except requests.exceptions.ReadTimeout:
        err = "timeout"
        set_cached(cache_dir, url, status=None, ok=False, text=None, error=err)
        FETCH_STATS["net_fail"] += 1
        return None, None, err
    except Exception as e:
        err = f"fetch_error: {type(e).__name__}"
        set_cached(cache_dir, url, status=None, ok=False, text=None, error=err)
        FETCH_STATS["net_fail"] += 1
        return None, None, err