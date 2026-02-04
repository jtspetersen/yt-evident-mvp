# app/tools/url_cache.py
import os
import json
import hashlib
from datetime import datetime, timedelta, timezone

def _now_utc():
    return datetime.now(timezone.utc)

def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def _path_for_url(cache_dir: str, url: str) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"{_sha256(url)}.json")

def get_cached(cache_dir: str, url: str, ttl_days: int) -> dict | None:
    """
    Returns cached record if present and not expired, else None.
    Record shape:
      {"url":..., "fetched_at": ISO8601, "status": int|None, "ok": bool, "text": str|None, "error": str|None}
    """
    path = _path_for_url(cache_dir, url)
    if not os.path.exists(path):
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            rec = json.load(f)
        fetched_at = rec.get("fetched_at")
        if not fetched_at:
            return None
        dt = datetime.fromisoformat(fetched_at.replace("Z", "+00:00"))
        if _now_utc() - dt > timedelta(days=int(ttl_days)):
            return None
        return rec
    except Exception:
        return None

def set_cached(cache_dir: str, url: str, status: int | None, ok: bool, text: str | None, error: str | None):
    path = _path_for_url(cache_dir, url)
    rec = {
        "url": url,
        "fetched_at": _now_utc().isoformat().replace("+00:00", "Z"),
        "status": status,
        "ok": bool(ok),
        "text": text,
        "error": error,
    }
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False)
    os.replace(tmp, path)
