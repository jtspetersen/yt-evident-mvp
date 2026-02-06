# app/store/creator_rollup.py
import os
import json
from collections import defaultdict, Counter

def load_creator_events(path: str = os.path.join("store", "creator_profiles.jsonl")):
    if not os.path.exists(path):
        return []
    events = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except Exception:
                continue
    return events

def rollup_by_channel(events):
    by = defaultdict(list)
    for e in events:
        by[(e.get("channel") or "Unknown").strip()].append(e)
    return by

def summarize_channel(events_for_channel: list, recent_n: int = 5):
    runs = len(events_for_channel)
    red_flag_counter = Counter()
    topic_counter = Counter()
    verdict_counter = Counter()

    for e in events_for_channel:
        for rf in (e.get("red_flags") or []):
            if isinstance(rf, str) and rf.strip():
                red_flag_counter[rf.strip()] += 1
        for t in (e.get("topics") or []):
            if isinstance(t, str) and t.strip():
                topic_counter[t.strip()] += 1
        vc = e.get("verdict_counts") or {}
        if isinstance(vc, dict):
            for k, v in vc.items():
                try:
                    verdict_counter[str(k)] += int(v)
                except Exception:
                    pass

    # recent runs (sorted by timestamp if present; else run_id)
    def sort_key(e):
        return (e.get("timestamp") or "", e.get("run_id") or "")

    recent = sorted(events_for_channel, key=sort_key, reverse=True)[:recent_n]

    return {
        "runs": runs,
        "top_red_flags": red_flag_counter.most_common(8),
        "top_topics": topic_counter.most_common(8),
        "verdict_totals": dict(verdict_counter),
        "recent": [
            {
                "timestamp": r.get("timestamp"),
                "run_id": r.get("run_id"),
                "input_file": r.get("input_file"),
                "outdir": r.get("outdir"),
            } for r in recent
        ]
    }
