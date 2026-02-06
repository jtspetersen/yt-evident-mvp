# CLAUDE.md — Project Context for Claude Code

## Project Overview

Evident Video Fact Checker is a Python-based fact-checking pipeline for YouTube transcripts. It extracts claims from video transcripts, searches for evidence via web sources, verifies claims using LLM reasoning, and generates fact-check reports with video scripts.

## Tech Stack

- **Language:** Python 3.11+
- **Data validation:** Pydantic 2.0+
- **HTTP:** requests
- **HTML parsing:** BeautifulSoup4 + lxml
- **Config:** PyYAML
- **LLM backend:** Ollama (local, accessed via REST API)
- **Search backend:** SearX (local metasearch instance)
- **No web framework** — CLI-only pipeline

## Project Structure

```
app/
├── main.py              # CLI entry point, 7-stage pipeline orchestration
├── policy.py            # Fact-checking standards, source hierarchy, voice guidelines
├── rollup.py            # Creator profile analytics
├── version.py           # Version tracking (placeholder)
├── pipeline/            # Core processing stages (ordered)
│   ├── ingest.py        # Normalize transcripts to JSON
│   ├── extract_claims.py # LLM-based claim extraction
│   ├── retrieve_evidence.py # Web search + evidence gathering
│   ├── verify_claims.py # LLM-based claim verification
│   ├── scorecard.py     # Rating aggregation (0-100 score)
│   └── write_outputs.py # Outline + script generation
├── schemas/             # Pydantic data models
│   ├── claim.py         # Claim model with ClaimType enum
│   └── verdict.py       # Verdict model with Rating enum
├── store/               # Persistent data (JSONL append-only)
│   ├── run_index.py     # Run history index
│   ├── creator_profiles.py # Per-channel profile events
│   └── creator_rollup.py   # Channel summary analytics
└── tools/               # Utilities
    ├── ollama_client.py  # LLM API wrapper with retry logic
    ├── searx.py          # Metasearch integration
    ├── fetch.py          # HTTP fetching with disk cache
    ├── url_cache.py      # Cache management with TTL
    ├── parse.py          # HTML to text extraction
    ├── logger.py         # Run logging
    ├── json_extract.py   # LLM output JSON parsing
    ├── snippets.py       # Evidence snippet generation
    └── review.py         # Interactive claim review
```

## How to Run

```bash
# Activate venv and run
./run.sh

# Or directly
python -m app.main --infile inbox/transcript.txt --channel "ChannelName"

# With interactive review mode
python -m app.main --infile inbox/transcript.txt --review
```

## Key Configuration

All config lives in `config.yaml`:
- `ollama.base_url` — Local Ollama API (default: http://localhost:11434)
- `ollama.model_extract/verify/write` — Models per pipeline stage
- `searx.base_url` — Local SearX instance (default: http://localhost:8080)
- `budgets.*` — Rate limits (max claims, sources, fetches)
- `cache.url_cache_days` — URL cache TTL (default: 7 days)

## Pipeline Stages (in order)

1. **normalize_transcript** — Text to JSON segments
2. **extract_claims** — LLM extracts checkable claims
3. **review_claims_interactive** — (optional) User edits/drops claims
4. **retrieve_evidence** — SearX search, URL fetch, HTML parse, snippet extract
5. **verify_claims** — LLM verifies each claim against evidence
6. **scorecard** — Aggregate verdicts to 0-100 score
7. **write_outline_and_script** — Generate review video outline + script

## Output

Each run creates a timestamped directory in `runs/` with numbered artifacts:
`00_transcript.raw.txt` through `07_summary.md`, plus `run.json` manifest and `run.log`.

## Development Notes

- No test suite exists yet
- No linter/formatter configured
- Dependencies in `Requirements.txt` (note: capital R)
- Runtime data dirs (`cache/`, `logs/`, `runs/`, `store/`, `inbox/`) are gitignored
- Windows environment (MSYS/Git Bash compatible)
- The `store/` top-level dir holds runtime JSONL data; `app/store/` holds the Python modules
