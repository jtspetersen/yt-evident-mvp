# Evident Roadmap

Four focus areas for improving pipeline quality and consistency.

---

## 1. Claim Extraction Quality

**Problem:** Many extracted "claims" are fragments of a larger contextual statement with multiple parts. The claim text often loses the context needed to verify it properly. For example, a causal argument like *"Because X happened, Y is now true"* might get split into two separate claims that each lack meaning on their own.

**Problem:** Extraction results vary significantly between runs on the same transcript. The LLM doesn't need to be perfect, but the current inconsistency undermines trust in the pipeline.

### 1a. Improve LLM contextual extraction

**Current state:** The extraction prompt (`extract_claims.py`) already instructs the LLM to combine dependent claims and make them self-contained. In practice, the model (qwen3:8b) frequently ignores this, especially with dense transcripts.

**Approach:**

- **Two-pass extraction.** First pass: extract a broad list of candidate claims at temperature 0.1. Second pass: feed the candidates back to the LLM with the surrounding transcript context and ask it to merge fragments, remove duplicates, and ensure each claim is self-contained. The second pass uses a simpler, more constrained prompt focused only on consolidation.

- **Structured context window.** Instead of passing raw segment text, pass each chunk as a numbered argument outline:
  ```
  Argument 1: [segments 3-7] Speaker claims X because Y, citing Z.
  Argument 2: [segments 8-12] Speaker asserts A, supported by B.
  ```
  This pre-structuring helps the LLM see argument boundaries rather than treating each sentence independently.

- **Claim completeness validation.** After extraction, run a lightweight check: for each claim, ask the LLM *"Can this claim be verified without any additional context? Yes/No. If no, what context is missing?"* Claims that fail get enriched with their missing context from the transcript.

- **Increase chunk overlap.** Current overlap is 5 segments (25% of chunk size 20). Increase to 8-10 segments (40-50%) to ensure argument boundaries are less likely to be split across chunks. This increases LLM calls but significantly reduces fragmentation.

### 1b. Human-in-the-loop claim grouping

**Current state:** The web UI review step allows keep/drop/edit on individual claims. There is no way to group or merge claims.

**Approach:**

- **Claim grouping UI.** Add a "Group" action to the review interface. Users can select 2+ claims and merge them into a claim group. The group gets a combined claim text (editable) and carries all original quotes. During verification, the group is verified as a single unit with all quotes as context.

- **Suggested groups.** Before showing claims for review, run a similarity pass (embedding-based or LLM-based) to suggest likely groups. Present these as *"These claims may be related — group them?"* checkboxes.

- **Claim schema change.** Add an optional `group_id` field to the Claim model. Claims sharing a `group_id` are verified together. The `claim_text` of the group becomes the combined, context-rich version.

### 1c. Improve extraction consistency

**Approach:**

- **Lower temperature to 0.0** for extraction. The current 0.1 introduces unnecessary variance. Claim extraction is a structured task — determinism is more valuable than diversity here.

- **Seed-based stability.** Pass a deterministic seed derived from the transcript hash to the Ollama API (if supported by the model). This ensures identical input produces identical output across runs.

- **Anchor extraction to quotes.** Restructure the prompt to require the LLM to first identify exact quotes from the transcript, then derive claims from those quotes. Quotes are deterministic anchors — if the model finds the same quotes, it will derive similar claims.

- **Post-extraction normalization.** After extraction, normalize claim text: standardize number formats, expand abbreviations, resolve pronouns using transcript context. This reduces surface-level variation between runs.

- **Ensemble extraction.** Run extraction 2-3 times and take the intersection (claims that appear in at least 2 out of 3 runs, using semantic similarity). More expensive but dramatically more consistent. Could be an optional `--stable` flag.

---

## 2. Verification Efficiency

**Problem:** Verification is the slowest and most resource-intensive stage. Two avenues to improve: better evidence sourcing and better model selection.

### 2a. Improve evidence sourcing

**Current state:** Search queries use the raw `claim_text` as the search query. Snippet relevance is scored by token overlap (bag-of-words). Sources are ranked by domain tier only.

**Approach:**

- **Multi-query strategy.** For each claim, generate 2-3 search queries instead of one:
  1. The claim text itself (current behavior)
  2. An entity-focused query: extract key entities (people, organizations, numbers, dates) and search for them specifically
  3. A negation query: search for the opposite of the claim to find contradicting evidence

  This triples search coverage without tripling fetch budget (many results will overlap).

- **Query reformulation.** Use a lightweight LLM call (or the extract model) to rewrite the claim as a search-engine-friendly query. Claims like *"The speaker asserts that inflation reached 9% in 2023"* become *"US inflation rate 2023 9%"*. Shorter, keyword-rich queries get better search results.

- **Source pre-filtering.** Before fetching full page content, check the search result title and snippet preview for relevance. Skip results where the preview has zero keyword overlap with the claim. This saves fetch budget for genuinely relevant pages.

- **Snippet scoring upgrade.** Replace token overlap with BM25 or TF-IDF scoring. Token overlap treats all words equally — BM25 weights rare, informative terms higher. Implementation: use `rank_bm25` package or a simple TF-IDF with scikit-learn.

- **Fact-check source priority.** When searching, append a second query targeting known fact-check databases: `site:factcheck.org OR site:politifact.com OR site:snopes.com "{key phrase}"`. These sources often have pre-verified verdicts that can anchor the LLM's reasoning.

### 2b. Better model matching for verification

**Current state:** Verification uses qwen3:30b (MoE, ~3B active parameters) at temperature 0.0. Each claim takes 30-90 seconds depending on evidence volume.

**Approach:**

- **Tiered verification.** Not all claims need the same model:
  - **Simple factual claims** (statistics, dates, named events): Use a smaller, faster model (qwen3:8b). These have clear right/wrong answers and don't need deep reasoning.
  - **Complex claims** (causal, medical/science, policy): Use the full model (qwen3:30b or larger). These require nuanced reasoning about evidence quality and rhetorical context.
  - **Claim type routing:** Use the `claim_type` field (already extracted) to route: `statistic`, `event_date`, `biography` → fast model; `causal`, `medical_science`, `policy_legal` → full model.

- **Evidence-gated skipping.** If a claim has zero relevant snippets after retrieval, skip the full LLM verification and directly assign `INSUFFICIENT EVIDENCE` with a canned explanation. The current code already does this (`verify_one` fast-fail) but could be more aggressive — if all snippets have relevance scores below a threshold (e.g., 0.2), treat as insufficient.

- **Batch verification.** Instead of verifying one claim at a time, batch 2-3 simple claims into a single LLM call with clear separators. This amortizes the model loading overhead. Only works for simple claims where cross-contamination risk is low.

- **Model recommendations by hardware:**

  | VRAM | Fast model (simple claims) | Full model (complex claims) |
  |------|---------------------------|----------------------------|
  | 24GB+ | qwen3:8b | qwen3:30b or deepseek-r1:32b |
  | 12-16GB | qwen3:4b | qwen3:14b |
  | 8GB | qwen3:1.7b | qwen3:8b |

---

## 3. Summary Template Consistency

**Problem:** The summary output (07_summary.md) varies in structure between runs. The narrative script sometimes omits sections, changes heading levels, or reorganizes content. The executive overview and narrative should follow a rigid template every run.

### 3a. Harden the report template

**Current state:** The report is assembled from deterministic sections (executive overview, claim details, sources) and LLM-generated sections (key findings, narrative script). The deterministic sections are consistent; the LLM sections vary.

**Approach:**

- **Deterministic executive overview.** The executive overview should be entirely template-driven with no LLM involvement:
  ```markdown
  # Evident Fact-Check Report: {channel}

  ## Executive Overview

  **{total_claims} claims analyzed** from "{transcript_title}"

  | Rating | Count |
  |--------|-------|
  | VERIFIED | {n} |
  | ... | ... |

  **Key concern areas:** {top 2-3 red flags or "None detected"}

  **Source quality:** {n} sources across tiers {tier_list}. {pct}% from tiers 1-3.

  {1-paragraph summary — LLM-generated but constrained to 3 sentences max}
  ```

- **Rigid narrative script template.** Instead of asking the LLM to "write a script with these sections," provide the exact skeleton and ask the LLM to fill in each section:
  ```
  Fill in each section below. Do not add, remove, or rename any sections.
  Do not change the heading levels. Write in the second person ("you'll find...").

  ## Introduction
  [2-3 sentences: what video, what channel, how many claims checked, overall pattern]

  ## Analysis
  ### High-Severity Findings
  [Walk through FALSE and LIKELY FALSE claims with quotes and evidence]

  ### Confirmed Claims
  [Walk through VERIFIED and LIKELY TRUE claims briefly]

  ### Unresolved Claims
  [Walk through INSUFFICIENT EVIDENCE and CONFLICTING EVIDENCE claims]

  ## Detected Patterns
  [Identify 2-4 recurring patterns: rhetorical tactics, source quality issues, topic areas]

  ## Conclusion
  [3-4 sentences: what's true, what's false, what viewers should take away]
  ```

- **Section-by-section generation.** Instead of one large LLM call for the entire script, generate each section independently:
  1. Introduction (input: channel, claim count, verdict summary)
  2. Analysis (input: sorted verdicts with evidence)
  3. Detected Patterns (input: red_flags, rhetorical_issues aggregated)
  4. Conclusion (input: verdict counts, key findings)

  This prevents the model from drifting or restructuring. Each call is shorter, more focused, and more consistent. If one section fails, only that section needs retry.

- **Post-generation validation.** After the LLM generates the script, validate that all required sections exist (check for `## Introduction`, `## Analysis`, `## Detected Patterns`, `## Conclusion`). If any are missing, re-generate only that section.

### 3b. Verbose readout section

**Current state:** Claim details, red flags, and sources are already deterministic and template-driven. They live in the same document as the narrative.

**Approach:**

- **Clear separation.** Structure the final report with explicit separators:
  ```
  PART 1: Executive Overview (template + 1-paragraph LLM summary)
  PART 2: Narrative Script (section-by-section LLM generation)
  PART 3: Detailed Findings (fully deterministic)
    - Claim Details (per-claim verdict, citations, flags)
    - Red Flags Summary
    - Sources Appendix
  ```

- **Key findings as structured data.** Instead of asking the LLM for free-form bullets, compute key findings deterministically:
  - Any FALSE claim with high severity → key finding
  - Any claim with rhetorical_issues → key finding
  - Any pattern where 3+ claims share the same red flag → key finding
  - Fall back to LLM-generated findings only if the deterministic list has fewer than 3 items

---

## Priority Order

| Priority | Area | Impact | Effort |
|----------|------|--------|--------|
| 1 | **3a. Harden summary template** | High — directly visible to users | Low-Medium |
| 2 | **1c. Extraction consistency** | High — foundational for everything downstream | Medium |
| 3 | **2a. Better evidence sourcing** | High — better sources = better verdicts | Medium |
| 4 | **1a. Two-pass extraction** | Medium — reduces claim fragments | Medium |
| 5 | **2b. Tiered model routing** | Medium — faster runs, same quality | Low-Medium |
| 6 | **3b. Verbose readout structure** | Low — mostly cosmetic reorganization | Low |
| 7 | **1b. Claim grouping UI** | Medium — powerful but complex UI work | High |
