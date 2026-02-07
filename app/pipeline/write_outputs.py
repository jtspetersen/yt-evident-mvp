# app/pipeline/write_outputs.py
import json
from app.tools.ollama_client import ollama_chat
from app.policy import VOICE_TONE_GUIDE


# ---------------------------------------------------------------------------
# Shared prompt fragments
# ---------------------------------------------------------------------------

_HARD_RULES = """\
HARD RULES:
- Calm confidence; no outrage, no dunking, no sarcasm.
- Evidence over ideology; no partisan framing language.
- Use ONLY transcript quotes and the verdicts provided.
- Do not introduce new facts.
- Output ONLY the section content. No headings, no code fences."""

_RHETORICAL_RULES = """\
CRITICAL — Identify rhetorical manipulation:
- When a TRUE fact is used to support a FALSE conclusion
- When causation is claimed without evidence (correlation ≠ causation)
- When statistics are cherry-picked to mislead
- When context is omitted to change meaning
- When fear, dichotomies, or other manipulation tactics are used

For these cases, acknowledge the fact is true but explain the misleading use:
"While [fact] is accurate, the speaker uses it to suggest [false conclusion],
which is not supported by evidence..."

Check each verdict's rhetorical_issues field and clearly explain any manipulation detected."""


# ---------------------------------------------------------------------------
# Section-specific system prompts
# ---------------------------------------------------------------------------

SUMMARY_SYSTEM = f"""\
You are an analyst writing a brief summary of fact-check results.

{_HARD_RULES}

Summarize these fact-check results in exactly 3 sentences. Be direct and factual.
State the overall pattern, the most significant finding, and what it means for viewers.

{VOICE_TONE_GUIDE}"""

INTRO_SYSTEM = f"""\
You are the Writer for a YouTube fact-check review channel.

{_HARD_RULES}

Write 2-3 sentences introducing this fact-check. State the channel name, how many
claims were checked, and the overall pattern of results (e.g. "most claims held up"
or "several key claims are unsupported").

{VOICE_TONE_GUIDE}"""

ANALYSIS_SYSTEM = f"""\
You are the Writer for a YouTube fact-check review channel.

{_HARD_RULES}

Write the analysis for a fact-check video script. The claims have been pre-sorted
into three groups for you. Write content for ALL THREE groups, clearly separated.

For each group, write about every claim or narrative group in that group:
- State what was said (use direct quotes from the transcript when available)
- Explain what the evidence shows
- Give the rating and why
- Call out any rhetorical issues or manipulation
- For narrative groups (GXXX entries), analyze the overall narrative pattern
- End each finding paragraph with a cross-reference: "**See [ID] in Claim Details.**"
  where [ID] is the claim ID (e.g. C002) or "narrative group G001" for groups.

{_RHETORICAL_RULES}

Format your output as three labeled sections using these EXACT markers:

===HIGH-SEVERITY===
[Content about FALSE and LIKELY FALSE claims here, plus any MISLEADING narrative groups]

===CONFIRMED===
[Content about TRUE and LIKELY TRUE claims here]

===UNRESOLVED===
[Content about INSUFFICIENT EVIDENCE and CONFLICTING EVIDENCE claims here]

If a group has no claims, write "No claims in this category." for that group.

{VOICE_TONE_GUIDE}"""

PATTERNS_SYSTEM = f"""\
You are the Writer for a YouTube fact-check review channel.

{_HARD_RULES}

Identify 2-4 recurring patterns from these fact-check results. Look for:
inflated numbers, misleading framing, cherry-picked data, omitted context,
fear-based rhetoric, correlation-as-causation, etc.

Write 1-2 sentences per pattern. If no clear patterns exist, state that plainly.

{VOICE_TONE_GUIDE}"""

CONCLUSION_SYSTEM = f"""\
You are the Writer for a YouTube fact-check review channel.

{_HARD_RULES}

Write a 3-4 sentence conclusion for this fact-check. Summarize:
- What's true and well-supported
- What's false or misleading
- What the audience should take away

{VOICE_TONE_GUIDE}"""


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


def _build_executive_overview(channel, counts, red_flags, tiers, total_claims,
                              summary_paragraph="", num_groups=0):
    """Build the top-of-report summary from scorecard data."""
    claims_line = f"{total_claims} claims analyzed"
    if num_groups:
        claims_line += f" | {num_groups} narrative cluster{'s' if num_groups != 1 else ''} detected"

    lines = [
        f"# Evident Fact-Check Report: {channel}",
        "",
        "## Executive Overview",
        "",
        claims_line,
        "",
        "| Rating | Count |",
        "|--------|-------|",
    ]
    for rating in ("TRUE", "LIKELY TRUE", "INSUFFICIENT EVIDENCE",
                    "CONFLICTING EVIDENCE", "LIKELY FALSE", "FALSE"):
        lines.append(f"| {rating} | {counts.get(rating, 0)} |")

    lines.append("")

    # Key concern areas (top 2-3 red flags)
    if red_flags:
        top_flags = sorted(red_flags.items(), key=lambda x: x[1], reverse=True)[:3]
        flags_str = ", ".join(f"{k} ({v})" for k, v in top_flags)
        lines.append(f"**Key concern areas:** {flags_str}")
    else:
        lines.append("**Key concern areas:** None detected")

    # Source quality stats
    if tiers:
        total_sources = sum(tiers.values())
        high_quality = sum(tiers.get(t, 0) for t in (1, 2, 3))
        pct = int(100 * high_quality / total_sources) if total_sources else 0
        tier_list = ", ".join(str(t) for t in sorted(tiers.keys()))
        lines.append(
            f"**Source quality:** {total_sources} sources across tiers {tier_list}. "
            f"{pct}% from tiers 1-3."
        )

    # Optional LLM summary paragraph
    if summary_paragraph:
        lines.append("")
        lines.append(summary_paragraph)

    return "\n".join(lines)


def _group_severity(gv):
    """Derive severity for a narrative group from its rating."""
    if gv is None:
        return "medium"
    if gv.narrative_rating in ("LARGELY MISLEADING", "MISLEADING", "UNSUPPORTED"):
        return "high"
    elif gv.narrative_rating == "PARTIALLY SUPPORTED":
        return "medium"
    return "low"


def _compute_key_findings(verdicts, red_flags, groups=None, group_verdicts=None):
    """Compute key findings deterministically from verdict data."""
    groups = groups or []
    group_verdicts = group_verdicts or []

    # Determine which claim IDs are inside groups (so we can skip them)
    grouped_claim_ids = set()
    for g in groups:
        grouped_claim_ids.update(g.claim_ids)

    gv_by_id = {gv.group_id: gv for gv in group_verdicts}

    lines = []
    seen_ids = set()

    # Collect high-severity FALSE/LIKELY FALSE individual claims (ungrouped)
    false_high = [v for v in verdicts
                  if v.rating in ("FALSE", "LIKELY FALSE")
                  and v.severity == "high"
                  and v.claim_id not in grouped_claim_ids]

    # Collect high-severity group verdicts
    high_groups = [(g, gv_by_id[g.group_id])
                   for g in groups
                   if g.group_id in gv_by_id
                   and _group_severity(gv_by_id[g.group_id]) == "high"]

    if false_high or high_groups:
        lines.append("**FALSE (high severity):**")
        for v in false_high:
            explanation = v.explanation[:150]
            if len(v.explanation) > 150:
                explanation += "..."
            lines.append(f"- **{v.claim_id}** — {explanation}")
            seen_ids.add(v.claim_id)

        for g, gv in high_groups:
            explanation = gv.explanation[:150]
            if len(gv.explanation) > 150:
                explanation += "..."
            lines.append(
                f"- **{g.group_id}** — Narrative group {g.narrative_thesis[:80]}: "
                f"{explanation}"
            )
            seen_ids.add(g.group_id)

    # Rhetorical issues (skip already listed and grouped claims)
    for v in verdicts:
        if (v.rhetorical_issues
                and v.claim_id not in seen_ids
                and v.claim_id not in grouped_claim_ids):
            issues = ", ".join(v.rhetorical_issues)
            lines.append(
                f"**Rhetorical issue in \"{v.claim_id}\":** {issues}"
            )
            seen_ids.add(v.claim_id)

    # Red flag patterns (3+ claims share same flag)
    for flag, count in red_flags.items():
        if count >= 3:
            lines.append(
                f"**Recurring pattern**: \"{flag}\" detected across {count} claims"
            )

    if not lines:
        lines.append("No high-severity findings detected.")

    return lines[:10]


def _render_individual_claim(v, claim):
    """Render a single individual claim detail entry as markdown lines."""
    claim_text = claim.claim_text if claim else v.claim_id

    lines = [
        f"#### {v.claim_id}: \"{claim_text}\"",
        "",
        f"**Rating:** {v.rating} | **Confidence:** {v.confidence:.0%} | **Severity:** {v.severity}",
        "",
        f"**Explanation:** {v.explanation}",
    ]

    # Source tiers
    if v.source_tiers_used:
        tier_counts = {}
        for t in v.source_tiers_used:
            tier_counts[t] = tier_counts.get(t, 0) + 1
        tiers_str = ", ".join(
            f"Tier {t} — {TIER_LABELS.get(t, 'Other')} ({n})"
            for t, n in sorted(tier_counts.items())
        )
        lines.append("")
        lines.append(f"**Sources:** {tiers_str}")

    # Citations
    if v.citations:
        lines.append("")
        lines.append("**Citations:**")
        for cit in v.citations:
            quote_preview = (cit.quote[:120] + "...") if len(cit.quote) > 120 else cit.quote
            lines.append(f"- [{cit.url}]({cit.url}) — \"{quote_preview}\"")

    # Red flags
    lines.append("")
    if v.red_flags:
        lines.append(f"**Red Flags:** {', '.join(v.red_flags)}")
    else:
        lines.append("**Red Flags:** None")

    # Rhetorical issues
    lines.append("")
    if v.rhetorical_issues:
        lines.append(f"**Rhetorical Issues:** {', '.join(v.rhetorical_issues)}")
    else:
        lines.append("**Rhetorical Issues:** None")

    # Corrected claim
    if v.corrected_claim:
        lines.append("")
        lines.append(f"**Corrected Claim:** {v.corrected_claim}")

    return lines


def _render_group_claim(g, gv, claims_by_id, verdict_by_id):
    """Render a narrative group detail entry as markdown lines."""
    lines = [
        f"#### {g.group_id}: \"{g.narrative_thesis}\"",
        "",
    ]

    if gv:
        severity = _group_severity(gv)
        lines.append(
            f"**Type:** NARRATIVE GROUP | **Rating:** {gv.narrative_rating} "
            f"| **Confidence:** {gv.narrative_confidence:.0%} | **Severity:** {severity}"
        )
        lines.append("")
        lines.append(f"**Explanation:** {gv.explanation}")

        # Component claims
        lines.append("")
        lines.append("**Component Claims:**")
        for cid in g.claim_ids:
            claim = claims_by_id.get(cid)
            v = verdict_by_id.get(cid)
            if claim:
                rating_str = v.rating if v else "Not verified"
                conf_str = f" ({v.confidence:.0%})" if v else ""
                lines.append(f"- **{cid} [{rating_str}{conf_str}]** — \"{claim.claim_text}\"")
            else:
                lines.append(f"- **{cid}**: *(claim not found)*")

        if gv.reasoning_gap:
            lines.append("")
            lines.append(f"**Reasoning Gap:** {gv.reasoning_gap}")

        # Sources — reference component claims
        lines.append("")
        lines.append("**Sources:** Not directly listed in narrative analysis, see component claims")

        # Red flags from component claims
        component_flags = set()
        for cid in g.claim_ids:
            v = verdict_by_id.get(cid)
            if v and v.red_flags:
                component_flags.update(v.red_flags)
        lines.append("")
        if component_flags:
            lines.append(f"**Red Flags:** {', '.join(sorted(component_flags))}")
        else:
            lines.append("**Red Flags:** None")

        # Rhetorical issues
        lines.append("")
        if gv.rhetorical_issues:
            lines.append(f"**Rhetorical Issues:** {', '.join(gv.rhetorical_issues)}")
        else:
            lines.append("**Rhetorical Issues:** None")

        # Rhetorical strategy from group
        if g.rhetorical_strategy:
            lines.append("")
            lines.append(f"**Rhetorical Strategy:** {g.rhetorical_strategy}")

        # Corrected narrative from reasoning gap
        if gv.reasoning_gap:
            lines.append("")
            lines.append(f"**Corrected Narrative:** {gv.reasoning_gap}")
    else:
        lines.append(
            "**Type:** NARRATIVE GROUP | **Rating:** N/A "
            "| **Confidence:** N/A | **Severity:** medium"
        )
        lines.append("")
        lines.append("**Component Claims:**")
        for cid in g.claim_ids:
            claim = claims_by_id.get(cid)
            if claim:
                lines.append(f"- **{cid}** — \"{claim.claim_text}\"")

    return lines


def _build_claim_details(verdicts, claims, groups=None, group_verdicts=None):
    """Build unified claim details organized by severity.

    Narrative groups are rendered inline alongside individual claims.
    Component claims (those inside groups) are suppressed from individual listings.
    """
    groups = groups or []
    group_verdicts = group_verdicts or []

    claims_by_id = {c.claim_id: c for c in claims}
    verdict_by_id = {v.claim_id: v for v in verdicts}
    gv_by_id = {gv.group_id: gv for gv in group_verdicts}

    # Component claim IDs appear only within their group
    grouped_claim_ids = set()
    for g in groups:
        grouped_claim_ids.update(g.claim_ids)

    # Build unified list: (severity_rank, sort_key, type, data)
    SEVERITY_RANK = {"high": 0, "medium": 1, "low": 2}
    items = []

    # Individual (ungrouped) verdicts
    for v in verdicts:
        if v.claim_id not in grouped_claim_ids:
            rank = SEVERITY_RANK.get(v.severity, 1)
            items.append((rank, v.claim_id, "individual", v))

    # Group verdicts
    for g in groups:
        gv = gv_by_id.get(g.group_id)
        severity = _group_severity(gv)
        rank = SEVERITY_RANK.get(severity, 1)
        items.append((rank, g.group_id, "group", (g, gv)))

    # Sort by severity rank, then by ID
    items.sort(key=lambda x: (x[0], x[1]))

    if not items:
        return "\n".join(["", "---", "", "## Claim Details", "",
                          "*No claims to display.*"])

    sections = ["", "---", "", "## Claim Details"]

    severity_labels = {
        0: "High-Severity Claims",
        1: "Medium-Severity Claims",
        2: "Lower-Severity Claims",
    }

    current_severity = None
    first_in_section = True

    for rank, item_id, item_type, data in items:
        # New severity section
        if rank != current_severity:
            current_severity = rank
            first_in_section = True
            sections.extend(["", f"### {severity_labels.get(rank, 'Other Claims')}", ""])

        # Separator between items within the same section
        if not first_in_section:
            sections.extend(["", "---", ""])
        first_in_section = False

        if item_type == "individual":
            sections.extend(_render_individual_claim(data, claims_by_id.get(data.claim_id)))
        elif item_type == "group":
            g, gv = data
            sections.extend(_render_group_claim(g, gv, claims_by_id, verdict_by_id))

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
    groups=None,
    group_verdicts=None,
) -> str:
    """
    Build a structured fact-check report.

    Section order:
      Executive Overview → Key Findings → Introduction → Analysis
      (High-Severity / Confirmed / Unresolved / Detected Patterns)
      → Conclusion → Claim Details (by severity) → Sources
    """
    claims = claims or []
    groups = groups or []
    group_verdicts = group_verdicts or []

    # --- Parse scorecard data ---
    counts, red_flags, tiers = _parse_scorecard(scorecard_md)
    total_claims = sum(counts.values())

    # --- Compute grouped claim IDs ---
    grouped_claim_ids = set()
    for g in groups:
        grouped_claim_ids.update(g.claim_ids)

    # --- Deterministic sections ---
    key_findings = _compute_key_findings(
        verdicts, red_flags, groups=groups, group_verdicts=group_verdicts,
    )
    # Build key findings with proper spacing between sections
    kf_lines = []
    for f in key_findings:
        # Add blank line before non-bullet items that follow bullet items
        if not f.startswith("- ") and kf_lines and kf_lines[-1].startswith("- "):
            kf_lines.append("")
        kf_lines.append(f)
    key_findings_md = "\n".join(kf_lines)
    details = _build_claim_details(
        verdicts, claims, groups=groups, group_verdicts=group_verdicts,
    )
    sources = _build_sources_appendix(verdicts)

    # --- Verdict data for LLM calls ---
    claims_by_id = {c.claim_id: c for c in claims}
    verdict_summaries = []
    for v in verdicts:
        claim = claims_by_id.get(v.claim_id)
        claim_text = claim.claim_text if claim else v.claim_id
        verdict_summaries.append({
            "claim_id": v.claim_id,
            "claim_text": claim_text,
            "rating": v.rating,
            "severity": v.severity,
            "confidence": v.confidence,
            "explanation": v.explanation,
            "red_flags": v.red_flags,
            "rhetorical_issues": v.rhetorical_issues,
        })

    # --- Build group summaries for LLM analysis ---
    gv_by_id = {gv.group_id: gv for gv in group_verdicts}
    group_summaries = []
    for g in groups:
        gv = gv_by_id.get(g.group_id)
        if gv:
            group_summaries.append({
                "claim_id": g.group_id,
                "claim_text": g.narrative_thesis,
                "rating": gv.narrative_rating,
                "severity": _group_severity(gv),
                "confidence": gv.narrative_confidence,
                "explanation": gv.explanation,
                "red_flags": [],
                "rhetorical_issues": gv.rhetorical_issues,
                "type": "narrative_group",
                "component_claim_ids": g.claim_ids,
            })

    # --- LLM call 1: Executive summary paragraph (3 sentences) ---
    summary_paragraph = _generate_summary_paragraph(
        ollama_base, model, channel, counts, total_claims, verdict_summaries
    )

    executive = _build_executive_overview(
        channel, counts, red_flags, tiers, total_claims, summary_paragraph,
        num_groups=len(groups),
    )

    # --- LLM calls 2-5: Narrative script sections ---
    intro = _generate_section_introduction(
        ollama_base, model, channel, counts, total_claims
    )
    # Send ungrouped individual claims + group summaries to analysis LLM
    ungrouped_summaries = [vs for vs in verdict_summaries if vs["claim_id"] not in grouped_claim_ids]
    analysis = _generate_section_analysis(
        ollama_base, model, ungrouped_summaries, transcript_json,
        group_summaries=group_summaries,
    )
    patterns = _generate_section_patterns(
        ollama_base, model, red_flags, verdict_summaries
    )
    conclusion = _generate_section_conclusion(
        ollama_base, model, counts, key_findings, total_claims
    )

    # --- Assemble narrative with hardcoded headings ---
    narrative = _assemble_narrative(intro, analysis, patterns, conclusion)

    # --- Assemble final report ---
    report_parts = [
        executive,
        "",
        "---",
        "",
        "## Key Findings",
        "",
        key_findings_md,
        "",
        "---",
        "",
        narrative,
        details,
        sources,
        "",
    ]

    report = "\n".join(report_parts)
    return report


# ---------------------------------------------------------------------------
# LLM generation helpers — section-by-section
# ---------------------------------------------------------------------------

def _strip_fences(text):
    """Strip accidental markdown code fences from LLM output."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[-1].strip() == "```":
            lines = lines[1:-1]
        else:
            lines = lines[1:]
        text = "\n".join(lines).strip()
    return text


def _generate_summary_paragraph(ollama_base, model, channel, counts,
                                total_claims, verdict_summaries):
    """Generate a 3-sentence executive summary paragraph."""
    payload = json.dumps({
        "channel": channel,
        "total_claims": total_claims,
        "verdict_counts": counts,
        "verdicts": verdict_summaries[:10],
    }, ensure_ascii=False)

    try:
        raw = ollama_chat(
            ollama_base, model, SUMMARY_SYSTEM, payload,
            temperature=0.1,
            force_json=False,
            show_progress=True,
            num_predict=256,
        )
        text = _strip_fences(raw)
        if text and len(text) > 20:
            return text
    except Exception:
        pass
    return ""


def _generate_section_introduction(ollama_base, model, channel, counts,
                                   total_claims):
    """Generate the Introduction section (2-3 sentences)."""
    payload = json.dumps({
        "channel": channel,
        "total_claims": total_claims,
        "verdict_counts": counts,
    }, ensure_ascii=False)

    try:
        raw = ollama_chat(
            ollama_base, model, INTRO_SYSTEM, payload,
            temperature=0.2,
            force_json=False,
            show_progress=True,
            num_predict=512,
        )
        text = _strip_fences(raw)
        if text and len(text) > 20:
            return text
    except Exception:
        pass

    # Deterministic fallback
    verified = counts.get("TRUE", 0) + counts.get("LIKELY TRUE", 0)
    false_ct = counts.get("FALSE", 0) + counts.get("LIKELY FALSE", 0)
    return (
        f"We checked {total_claims} claims from {channel}. "
        f"{verified} were supported by evidence, while {false_ct} were not."
    )


def _generate_section_analysis(ollama_base, model, verdict_summaries,
                               transcript_json, group_summaries=None):
    """Generate the Analysis section with pre-sorted verdict groups."""
    group_summaries = group_summaries or []

    # Pre-sort verdicts into 3 groups
    high_severity = [v for v in verdict_summaries
                     if v["rating"] in ("FALSE", "LIKELY FALSE")]
    confirmed = [v for v in verdict_summaries
                 if v["rating"] in ("TRUE", "LIKELY TRUE")]
    unresolved = [v for v in verdict_summaries
                  if v["rating"] in ("INSUFFICIENT EVIDENCE", "CONFLICTING EVIDENCE")]

    # Add high-severity group verdicts to high_severity list
    for gs in group_summaries:
        if gs.get("severity") == "high":
            high_severity.append(gs)

    payload = json.dumps({
        "high_severity_claims": high_severity,
        "confirmed_claims": confirmed,
        "unresolved_claims": unresolved,
        "transcript_segments_sample": transcript_json["segments"][:30],
    }, ensure_ascii=False)

    try:
        raw = ollama_chat(
            ollama_base, model, ANALYSIS_SYSTEM, payload,
            temperature=0.3,
            force_json=False,
            show_progress=True,
            num_predict=3072,
        )
        text = _strip_fences(raw)
        if text and len(text) > 20:
            return _parse_analysis_sections(text)
    except Exception:
        pass

    # Deterministic fallback
    return {
        "high": "*(Analysis could not be generated — see Claim Details below)*",
        "confirmed": "*(See Claim Details below)*",
        "unresolved": "*(See Claim Details below)*",
    }


def _parse_analysis_sections(text):
    """Parse the ===MARKER=== delimited analysis into 3 sections."""
    sections = {"high": "", "confirmed": "", "unresolved": ""}

    # Try to split on markers
    if "===HIGH-SEVERITY===" in text:
        parts = text.split("===HIGH-SEVERITY===", 1)
        remainder = parts[1] if len(parts) > 1 else ""

        if "===CONFIRMED===" in remainder:
            high_part, remainder = remainder.split("===CONFIRMED===", 1)
            sections["high"] = high_part.strip()
        else:
            sections["high"] = remainder.strip()
            return sections

        if "===UNRESOLVED===" in remainder:
            confirmed_part, unresolved_part = remainder.split("===UNRESOLVED===", 1)
            sections["confirmed"] = confirmed_part.strip()
            sections["unresolved"] = unresolved_part.strip()
        else:
            sections["confirmed"] = remainder.strip()
    else:
        # Model didn't use markers — put everything in high-severity
        sections["high"] = text.strip()

    # Ensure no empty sections
    for key in sections:
        if not sections[key] or len(sections[key]) < 10:
            sections[key] = "No claims in this category."

    return sections


def _generate_section_patterns(ollama_base, model, red_flags, verdict_summaries):
    """Generate the Detected Patterns section."""
    # Aggregate rhetorical issues across all verdicts
    all_rhetorical = []
    for v in verdict_summaries:
        all_rhetorical.extend(v.get("rhetorical_issues", []))

    payload = json.dumps({
        "red_flags": red_flags,
        "rhetorical_issues_all": all_rhetorical,
        "verdict_summaries": [
            {"claim_id": v["claim_id"], "rating": v["rating"],
             "red_flags": v["red_flags"], "rhetorical_issues": v["rhetorical_issues"]}
            for v in verdict_summaries
        ],
    }, ensure_ascii=False)

    try:
        raw = ollama_chat(
            ollama_base, model, PATTERNS_SYSTEM, payload,
            temperature=0.2,
            force_json=False,
            show_progress=True,
            num_predict=1024,
        )
        text = _strip_fences(raw)
        if text and len(text) > 20:
            return text
    except Exception:
        pass

    # Deterministic fallback from red flags
    if red_flags:
        lines = []
        for flag, count in sorted(red_flags.items(), key=lambda x: x[1], reverse=True)[:4]:
            lines.append(f"- **{flag}**: detected in {count} claim(s)")
        return "\n".join(lines)

    return "No recurring patterns detected."


def _generate_section_conclusion(ollama_base, model, counts, key_findings,
                                 total_claims):
    """Generate the Conclusion section (3-4 sentences)."""
    payload = json.dumps({
        "total_claims": total_claims,
        "verdict_counts": counts,
        "key_findings": key_findings,
    }, ensure_ascii=False)

    try:
        raw = ollama_chat(
            ollama_base, model, CONCLUSION_SYSTEM, payload,
            temperature=0.2,
            force_json=False,
            show_progress=True,
            num_predict=512,
        )
        text = _strip_fences(raw)
        if text and len(text) > 20:
            return text
    except Exception:
        pass

    # Deterministic fallback
    verified = counts.get("TRUE", 0) + counts.get("LIKELY TRUE", 0)
    false_ct = counts.get("FALSE", 0) + counts.get("LIKELY FALSE", 0)
    return (
        f"Of {total_claims} claims checked, {verified} were supported by evidence "
        f"and {false_ct} were contradicted. Viewers should independently verify "
        f"key claims before drawing conclusions."
    )


def _assemble_narrative(intro, analysis, patterns, conclusion):
    """Assemble narrative script sections with hardcoded headings."""
    # analysis is a dict with keys: high, confirmed, unresolved
    if isinstance(analysis, str):
        analysis = {"high": analysis, "confirmed": "", "unresolved": ""}

    parts = [
        "## Introduction",
        "",
        intro,
        "",
        "---",
        "",
        "## Analysis",
        "",
        "### High-Severity Findings",
        "",
        analysis.get("high", "No claims in this category."),
        "",
        "### Confirmed Claims",
        "",
        analysis.get("confirmed", "No claims in this category."),
        "",
        "### Unresolved Claims",
        "",
        analysis.get("unresolved", "No claims in this category."),
        "",
        "### Detected Patterns",
        "",
        patterns,
        "",
        "---",
        "",
        "## Conclusion",
        "",
        conclusion,
    ]

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Scorecard parser
# ---------------------------------------------------------------------------

def _parse_scorecard(scorecard_md: str):
    """Extract counts, red_flags, and tiers from the scorecard markdown."""
    counts = {}
    red_flags = {}
    tiers = {}

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
        elif sec_stripped.startswith("Source tiers used"):
            try:
                tiers = json.loads(sec_stripped.split("\n", 1)[1].strip())
            except (json.JSONDecodeError, IndexError):
                pass

    return counts, red_flags, tiers


def write_text(path: str, text: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
