# app/tools/json_extract.py
import json
import re

def extract_json(text: str):
    """
    Extract first valid JSON object or array from a string.
    Handles cases where the model wraps JSON in prose or code fences.
    """
    # Strip common markdown fences
    cleaned = text.strip()
    cleaned = re.sub(r"^```(json)?", "", cleaned, flags=re.I).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()

    # Try direct parse first
    try:
        return json.loads(cleaned)
    except Exception:
        pass

    # Search for first {...} or [...] block
    match = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", cleaned)
    if not match:
        raise ValueError("No JSON found in model output")
    raw = match.group(1)

    return json.loads(raw)
