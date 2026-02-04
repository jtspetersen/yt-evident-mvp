# app/pipeline/ingest.py
import json
from datetime import datetime
from dateutil import tz

def normalize_transcript(text: str):
    paras = [p.strip() for p in text.split("\n") if p.strip()]
    segments = []
    for idx, p in enumerate(paras, start=1):
        segments.append({
            "id": f"S{idx:03d}",
            "timestamp": None,
            "speaker": None,
            "text": p
        })
    return {"video": {"title": None, "url": None, "channel": None}, "segments": segments}

def write_json(path: str, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def now_iso(tz_name: str):
    zone = tz.gettz(tz_name)
    return datetime.now(zone).isoformat()
