# app/pipeline/write_outputs.py
import json
from app.tools.ollama_client import ollama_chat
from app.policy import VOICE_TONE_GUIDE

WRITER_SYSTEM = f"""
You are the Writer for a YouTube review channel.

HARD RULES:
- Calm confidence; no outrage, no dunking, no sarcasm.
- Evidence over ideology; no partisan framing language.
- Use ONLY transcript quotes and the verdicts provided.
- Do not introduce new facts. No new claims.
- If something is UNCERTAIN, say so plainly.

CRITICAL: Identify rhetorical manipulation:
- When a TRUE fact is used to support a FALSE conclusion
- When causation is claimed without evidence (correlation ≠ causation)
- When statistics are cherry-picked to mislead
- When context is omitted to change meaning
- When fear, dichotomies, or other manipulation tactics are used

For these cases, acknowledge the fact is true but explain the misleading use:
"While [fact] is accurate, the speaker uses it to suggest [false conclusion],
which is not supported by evidence..."

Check each verdict's rhetorical_issues field and clearly explain any manipulation detected.

Output a single markdown document with:
1) Review Outline
2) Script Draft (spoken, clear, not long)
3) Citations Appendix: claim_id -> bullet list of URLs

{VOICE_TONE_GUIDE}
"""

def write_outline_and_script(ollama_base: str, model: str, transcript_json, verdicts, scorecard_md) -> str:
    payload = {
        "scorecard": scorecard_md,
        "verdicts": [v.model_dump() for v in verdicts],
        "transcript_segments_sample": transcript_json["segments"][:40],
    }
    raw = ollama_chat(ollama_base, model, WRITER_SYSTEM, json.dumps(payload, ensure_ascii=False), temperature=0.3)
    return raw

def write_text(path: str, text: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
