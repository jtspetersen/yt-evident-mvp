# app/pipeline/write_outputs.py
import json
from app.tools.ollama_client import ollama_chat
from app.tools.json_extract import extract_json
from app.policy import VOICE_TONE_GUIDE

# ---------------------------------------------------------------------------
# LLM prompts — two separate calls for reliability
# ---------------------------------------------------------------------------

FINDINGS_SYSTEM = """
You are an analyst summarising fact-check results.

Given a scorecard and list of verdicts, respond with a JSON object containing
exactly one key:

"key_findings" — an array of 3-5 short bullet strings summarising the most
important findings. Focus on: major falsehoods, notable confirmations, patterns
of rhetorical manipulation, and overall credibility assessment.

Example:
{
  "key_findings": [
    "The central claim about X is FALSE — no peer-reviewed evidence supports it.",
    "Statistics cited about Y are accurate but cherry-picked to omit Z.",
    "Most background facts check out; the core argument does not."
  ]
}
"""

SCRIPT_SYSTEM = f"""
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

{VOICE_TONE_GUIDE}

Write a conversational video script in markdown suitable for narration.
Structure it as:

## Introduction
A brief opening stating the video/channel being reviewed and the overall score.

## Analysis
Walk through the claims grouped by severity/impact. For each claim:
- State what was said (use direct quotes from the transcript)
- Explain what the evidence shows
- Give the rating and why
- Call out any rhetorical issues or manipulation

## Patterns
Identify recurring patterns: inflated numbers, misleading framing, cherry-picked
data, scapegoating, etc.

## Conclusion
Summarise the overall picture — what's true, what's false, and what the audience
should take away.

Output ONLY the markdown script. Do not wrap it in JSON or code fences.
"""


# ---------------------------------------------------------------------------
# Deterministic template builders
# ---------------------------------------------------------------------------

TIER_LABELS = {
    1: "Scholarly journals",
    2: "Academic institutions",
    3: "Government / intl orgs",
    4: "Research orgs / think tanks",
    5: "Established news agencies",
    6: "General web",
}


def _build_executive_overview(channel, overall, counts, red_flags):
    """Build the top-of-report summary from scorecard data."""
    lines = [
        f"# Evident Fact-Check Report: {channel}",
        "",
        "## Executive Overview",
        "",
        f"**Overall Truthfulness Score: {overall}/100**",
        "",
        "| Rating | Count |",
        "|--------|-------|",
    ]
    for rating in ("VERIFIED", "LIKELY TRUE", "UNCERTAIN", "LIKELY FALSE", "FALSE"):
        lines.append(f"| {rating} | {counts.get(rating, 0)} |")

    lines.append("")

    if red_flags:
        flags_str = ", ".join(f"{k} ({v})" for k, v in red_flags.items())
        lines.append(f"**Red Flags Detected:** {flags_str}")
    else:
        lines.append("**Red Flags Detected:** None")

    return "\n".join(lines)


def _build_claim_details(verdicts, claims):
    """Build per-claim detail sections from verdicts joined with claims."""
    claims_by_id = {c.claim_id: c for c in claims}
    sections = ["", "---", "", "## Claim Details"]

    for v in verdicts:
        claim = claims_by_id.get(v.claim_id)
        claim_text = claim.claim_text if claim else v.claim_id

        sections.append("")
        sections.append(f"### {v.claim_id}: \"{claim_text}\"")
        sections.append("")
        sections.append(
            f"- **Rating:** {v.rating} | **Confidence:** {v.confidence:.0%} "
            f"| **Severity:** {v.severity}"
        )
        sections.append(f"- **Explanation:** {v.explanation}")

        # Source tiers
        if v.source_tiers_used:
            tier_counts = {}
            for t in v.source_tiers_used:
                tier_counts[t] = tier_counts.get(t, 0) + 1
            tiers_str = ", ".join(
                f"Tier {t} — {TIER_LABELS.get(t, 'Other')} ({n})"
                for t, n in sorted(tier_counts.items())
            )
            sections.append(f"- **Sources:** {tiers_str}")

        # Citations
        if v.citations:
            sections.append("- **Citations:**")
            for cit in v.citations:
                quote_preview = (cit.quote[:120] + "...") if len(cit.quote) > 120 else cit.quote
                sections.append(f"  - [{cit.url}]({cit.url}) — \"{quote_preview}\"")

        # Red flags
        if v.red_flags:
            sections.append(f"- **Red Flags:** {', '.join(v.red_flags)}")
        else:
            sections.append("- **Red Flags:** None")

        # Rhetorical issues
        if v.rhetorical_issues:
            sections.append(f"- **Rhetorical Issues:** {', '.join(v.rhetorical_issues)}")
        else:
            sections.append("- **Rhetorical Issues:** None")

        # Corrected claim
        if v.corrected_claim:
            sections.append(f"- **Corrected Claim:** {v.corrected_claim}")

    return "\n".join(sections)


def _build_sources_appendix(verdicts):
    """Build a deduplicated sources list from all citation data."""
    seen = {}
    for v in verdicts:
        for cit in v.citations:
            if cit.url not in seen:
                seen[cit.url] = cit.tier

    if not seen:
        return "\n---\n\n## Sources\n\nNo citations available."

    lines = ["", "---", "", "## Sources", ""]
    for url, tier in sorted(seen.items(), key=lambda x: (x[1], x[0])):
        label = TIER_LABELS.get(tier, "Other")
        lines.append(f"- [{url}]({url}) — Tier {tier} ({label})")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def write_outline_and_script(
    ollama_base: str,
    model: str,
    transcript_json,
    verdicts,
    scorecard_md,
    claims=None,
    channel="Unknown Channel",
) -> str:
    """
    Build a structured fact-check report.

    Deterministic sections (executive overview, claim details, sources) are
    built in Python. Two separate LLM calls generate the key findings (JSON)
    and narrative script (plain markdown) for reliability.
    """
    # --- Parse score data from scorecard_md ---
    overall, counts, red_flags = _parse_scorecard(scorecard_md)

    # --- Deterministic sections ---
    executive = _build_executive_overview(channel, overall, counts, red_flags)
    details = _build_claim_details(verdicts, claims or [])
    sources = _build_sources_appendix(verdicts)

    # --- Shared payload for LLM calls ---
    payload = json.dumps({
        "scorecard": scorecard_md,
        "verdicts": [v.model_dump() for v in verdicts],
        "transcript_segments_sample": transcript_json["segments"][:40],
    }, ensure_ascii=False)

    # --- LLM call 1: Key findings (small JSON) ---
    key_findings_md = _generate_key_findings(ollama_base, model, payload)

    # --- LLM call 2: Narrative script (plain markdown) ---
    narrative_md = _generate_narrative_script(ollama_base, model, payload)

    # --- Assemble final report ---
    report = "\n".join([
        executive,
        "",
        "### Key Findings",
        "",
        key_findings_md,
        details,
        "",
        "---",
        "",
        "## Narrative Script",
        "",
        narrative_md,
        sources,
        "",
    ])

    return report


# ---------------------------------------------------------------------------
# LLM generation helpers
# ---------------------------------------------------------------------------

def _generate_key_findings(ollama_base, model, payload_json):
    """Generate 3-5 bullet key findings via a small JSON LLM call."""
    raw = ollama_chat(
        ollama_base, model, FINDINGS_SYSTEM, payload_json,
        temperature=0.2,
        force_json=True,
        show_progress=True,
        num_predict=1024,
    )

    try:
        data = extract_json(raw)
        findings = data.get("key_findings", [])
        if isinstance(findings, list) and findings:
            return "\n".join(f"- {f}" for f in findings)
    except Exception:
        pass

    return "- *(Key findings could not be generated)*"


def _generate_narrative_script(ollama_base, model, payload_json):
    """Generate the narrative video script as plain markdown."""
    raw = ollama_chat(
        ollama_base, model, SCRIPT_SYSTEM, payload_json,
        temperature=0.3,
        force_json=False,
        show_progress=True,
        num_predict=4096,
    )

    # Strip any accidental code fences the model might wrap around the output
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first line (```markdown or ```) and last line (```)
        if lines[-1].strip() == "```":
            lines = lines[1:-1]
        else:
            lines = lines[1:]
        text = "\n".join(lines).strip()

    if not text:
        return "*(No narrative script generated)*"

    return text


# ---------------------------------------------------------------------------
# Scorecard parser
# ---------------------------------------------------------------------------

def _parse_scorecard(scorecard_md: str):
    """Extract overall score, counts, and red_flags from the scorecard markdown."""
    overall = 0
    counts = {}
    red_flags = {}

    for line in scorecard_md.splitlines():
        if line.startswith("**Overall score:**"):
            try:
                overall = int(line.split("**")[2].strip().split("/")[0])
            except (IndexError, ValueError):
                pass

    # Extract JSON blocks from scorecard sections
    sections = scorecard_md.split("##")
    for sec in sections:
        sec_stripped = sec.strip()
        if sec_stripped.startswith("Verdict counts"):
            try:
                counts = json.loads(sec_stripped.split("\n", 1)[1].strip())
            except (json.JSONDecodeError, IndexError):
                pass
        elif sec_stripped.startswith("Red flags detected"):
            try:
                red_flags = json.loads(sec_stripped.split("\n", 1)[1].strip())
            except (json.JSONDecodeError, IndexError):
                pass

    return overall, counts, red_flags


def write_text(path: str, text: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
