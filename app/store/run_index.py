# app/store/run_index.py
import os
import json
from datetime import datetime

def append_run_index(
    run_id: str,
    input_file: str,
    outdir: str,
    verdict_counts: dict,
    duration_sec: float,
):
    os.makedirs("store", exist_ok=True)
    path = os.path.join("store", "run_index.jsonl")
    rec = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "input_file": input_file,
        "outdir": outdir,
        "verdict_counts": verdict_counts,
        "duration_sec": round(float(duration_sec), 2),
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
