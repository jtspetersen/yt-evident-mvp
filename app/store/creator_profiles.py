# app/store/creator_profiles.py
import os
import json
from datetime import datetime

def _ensure_store():
    os.makedirs("store", exist_ok=True)

def append_creator_profile_event(
    channel: str,
    run_id: str,
    verdict_counts: dict,
    red_flags: list,
    topics: list,
    input_file: str,
    outdir: str,
):
    """
    Append one JSONL line per run.
    This is intentionally append-only so you can compute trends later.
    """
    _ensure_store()
    path = os.path.join("store", "creator_profiles.jsonl")
    rec = {
        "timestamp": datetime.now().isoformat(),
        "channel": channel or "Unknown",
        "run_id": run_id,
        "verdict_counts": verdict_counts or {},
        "red_flags": red_flags or [],
        "topics": topics or [],
        "input_file": input_file,
        "outdir": outdir,
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
