# app/pipeline/ingest.py
import json
from datetime import datetime
from dateutil import tz

def normalize_transcript(text: str):
    import re

    lines = [line.strip() for line in text.split("\n") if line.strip()]
    segments = []

    current_text = []
    current_timestamp = None
    current_speaker = None

    for line in lines:
        # Skip lines that are just timestamps (e.g., "00:02")
        if re.match(r'^\d{1,2}:\d{2}$', line):
            continue

        # Parse timestamp and content (format: "00:00 text" or "00:00 >> text")
        match = re.match(r'^(\d{1,2}:\d{2})\s+(>>)?\s*(.*)$', line)

        if match:
            timestamp = match.group(1)
            speaker_marker = match.group(2)
            content = match.group(3).strip()

            # Detect speaker change or significant pause (10+ seconds)
            speaker_changed = speaker_marker is not None
            time_gap = False

            if current_timestamp and timestamp:
                try:
                    curr_secs = sum(int(x) * 60 ** i for i, x in enumerate(reversed(current_timestamp.split(":"))))
                    new_secs = sum(int(x) * 60 ** i for i, x in enumerate(reversed(timestamp.split(":"))))
                    time_gap = (new_secs - curr_secs) >= 10
                except:
                    pass

            # Flush current segment if speaker changed, time gap, or starts new sentence
            if current_text and (speaker_changed or time_gap):
                segments.append({
                    "id": f"S{len(segments)+1:03d}",
                    "timestamp": current_timestamp,
                    "speaker": "Speaker 2" if current_speaker else "Speaker 1",
                    "text": " ".join(current_text).strip()
                })
                current_text = []

            # Update state
            if speaker_changed:
                current_speaker = not current_speaker
            if not current_timestamp or speaker_changed or time_gap:
                current_timestamp = timestamp

            # Add content if not empty
            if content:
                current_text.append(content)
        else:
            # Line without timestamp - append to current segment
            if line:
                current_text.append(line)

    # Flush remaining text
    if current_text:
        segments.append({
            "id": f"S{len(segments)+1:03d}",
            "timestamp": current_timestamp,
            "speaker": "Speaker 2" if current_speaker else "Speaker 1",
            "text": " ".join(current_text).strip()
        })

    return {"video": {"title": None, "url": None, "channel": None}, "segments": segments}

def write_json(path: str, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def now_iso(tz_name: str):
    zone = tz.gettz(tz_name)
    return datetime.now(zone).isoformat()
