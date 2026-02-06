# app/tools/fetch.py
import os
import threading
import requests
from requests.adapters import HTTPAdapter

from app.tools.url_cache import get_cached, set_cached

# ---------------------------------------------------------------------------
# Module-level session with connection pooling
# ---------------------------------------------------------------------------
_SESSION = requests.Session()
_SESSION.headers["User-Agent"] = "Mozilla/5.0 (EvidentBot/0.1; +local)"
_SESSION.mount("https://", HTTPAdapter(pool_connections=12, pool_maxsize=12))
_SESSION.mount("http://", HTTPAdapter(pool_connections=4, pool_maxsize=4))

# ---------------------------------------------------------------------------
# Thread-safe fetch stats
# ---------------------------------------------------------------------------
_STATS_LOCK = threading.Lock()

FETCH_STATS = {
    "cache_hit_ok": 0,
    "cache_hit_fail": 0,
    "cache_miss": 0,
    "net_ok": 0,
    "net_fail": 0,
}

def _inc_stat(key: str):
    with _STATS_LOCK:
        FETCH_STATS[key] += 1

# ---------------------------------------------------------------------------
# Cached config read (parsed once per process)
# ---------------------------------------------------------------------------
_CACHE_DAYS = None

def _get_cache_days(default_days: int = 7) -> int:
    global _CACHE_DAYS
    if _CACHE_DAYS is not None:
        return _CACHE_DAYS
    try:
        import yaml
        if os.path.exists("config.yaml"):
            with open("config.yaml", "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            _CACHE_DAYS = int((cfg.get("cache", {}) or {}).get("url_cache_days", default_days))
            return _CACHE_DAYS
    except Exception:
        pass
    _CACHE_DAYS = int(default_days)
    return _CACHE_DAYS

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_url(url: str, timeout_sec: int = 20):
    """
    Returns (html_text | None, status_code | None, error | None)

    Uses disk cache:
      cache/url/<sha>.json
    TTL days from config.yaml cache.url_cache_days (default 7).
    Thread-safe — can be called from multiple threads concurrently.
    """
    cache_days = _get_cache_days(7)
    cache_dir = os.path.join("cache", "url")

    cached = get_cached(cache_dir, url, ttl_days=cache_days)
    if cached is not None:
        if cached.get("ok") and cached.get("text"):
            _inc_stat("cache_hit_ok")
            return cached.get("text"), cached.get("status"), None
        _inc_stat("cache_hit_fail")
        return None, cached.get("status"), cached.get("error") or "cached failure"

    _inc_stat("cache_miss")

    try:
        r = _SESSION.get(url, timeout=int(timeout_sec))
        status = r.status_code
        if 200 <= status < 300:
            text = r.text
            set_cached(cache_dir, url, status=status, ok=True, text=text, error=None)
            _inc_stat("net_ok")
            return text, status, None
        else:
            err = f"HTTP {status}"
            set_cached(cache_dir, url, status=status, ok=False, text=None, error=err)
            _inc_stat("net_fail")
            return None, status, err
    except requests.exceptions.ReadTimeout:
        err = "timeout"
        set_cached(cache_dir, url, status=None, ok=False, text=None, error=err)
        _inc_stat("net_fail")
        return None, None, err
    except Exception as e:
        err = f"fetch_error: {type(e).__name__}"
        set_cached(cache_dir, url, status=None, ok=False, text=None, error=err)
        _inc_stat("net_fail")
        return None, None, err
