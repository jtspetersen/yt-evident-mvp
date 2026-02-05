# YT Evident - Fact Checker MVP

A local fact-checking pipeline for analyzing video transcripts and verifying claims using evidence-based research.

## Features

### Core Capabilities
- **Transcript ingestion** - Handles fragmented YouTube transcripts with speaker detection
- **Claim extraction** - Overlapping chunks prevent missing claims at segment boundaries
- **Evidence retrieval** - 6-tier quality system prioritizes scholarly sources over forums/blogs
- **Claim verification** - LLM reasoning with citations, confidence scoring, and rhetorical analysis
- **Report generation** - Detailed verdicts, 0-100 truthfulness score, and video script outline

### Recent Improvements (2026-02-05)
- Overlapping chunks with deduplication
- Enhanced 6-tier source quality filtering (see Configuration)
- Complete, self-contained claim extraction
- Rhetorical manipulation detection (false causation, cherry-picking)

## Project Structure

```
yt-evident-mvp/
├── app/
│   ├── pipeline/      # Core processing stages
│   ├── schemas/       # Data models (Pydantic)
│   ├── store/         # Data persistence
│   └── tools/         # Utilities (fetch, parse, ollama client)
├── inbox/             # Input transcripts
├── runs/              # Output directories (timestamped)
├── config.yaml        # Configuration
├── run.sh             # Legacy run script
└── runvid             # Main run script (auto-activates venv)
```

## Setup

1. **Create virtual environment:**
   ```bash
   python -m venv venv
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure `config.yaml`** (see Configuration section below)

4. **Run the pipeline:**
   ```bash
   ./runvid
   ```

## Usage

The `runvid` script automatically activates the virtual environment and runs the pipeline.

### Command-Line Flags

| Flag | Description | Default |
|------|-------------|---------|
| `--infile <path>` | Path to transcript file (.txt or .md) | Newest file in `inbox/` |
| `--channel <name>` | Channel/creator name for this run | Inferred from filename |
| `--review` | Interactive mode: review/edit/drop claims before verification | Disabled |
| `--quiet` | Suppress DEBUG/INFO output (errors/warnings only) | Disabled |
| `--verbose` | Show all DEBUG output (overrides --quiet) | Disabled |

### Examples

```bash
# Basic: auto-detect newest file in inbox/
./runvid

# Specify file and channel
./runvid --infile inbox/transcript.txt --channel "Channel Name"

# Enable interactive claim review
./runvid --infile inbox/transcript.txt --review

# Debug mode (verbose logging)
./runvid --verbose

# Quiet mode (errors only)
./runvid --quiet

# Direct Python (no venv auto-activation)
python -m app.main --infile inbox/transcript.txt --review
```

## Requirements

- Python 3.11+
- Ollama (for LLM verification)
- SearX instance (for evidence retrieval)

## Configuration

The `config.yaml` file controls all pipeline behavior:

### Models
```yaml
ollama:
  model_extract: "qwen3:8b"      # Lightweight model for claim extraction
  model_verify: "qwen3:30b"      # Stronger model for verification
  model_write: "gemma3:27b"      # Creative model for script writing
```

### Search & Evidence Quality
```yaml
searx:
  deny_domains:                   # Excluded sources (forums, blogs, social media)
    - reddit.com
    - quora.com
    - wordpress.com
    # ... etc
```

The pipeline uses a 6-tier source quality system:
- **Tier 1**: Top scholarly journals (Nature, Science, NEJM, etc.)
- **Tier 2**: Academic institutions (.edu, .ac.uk)
- **Tier 3**: Government and international organizations (.gov, WHO, UN)
- **Tier 4**: Research organizations and think tanks (Pew Research, Brookings, etc.)
- **Tier 5**: Established news agencies (Reuters, AP, BBC, NPR)
- **Tier 6**: Everything else

### Budgets
```yaml
budgets:
  max_claims: 25                  # Maximum claims to extract per video
  max_sources_per_claim: 5        # Search results per claim
  max_fetches_per_run: 80         # Total URL fetches allowed
  fetch_timeout_sec: 25           # Timeout for each URL fetch
```

## Output

Each run creates a timestamped directory in `runs/` with the following artifacts:

```
runs/YYYYMMDD_HHMMSS__channel_name__video_title/
├── 00_transcript.raw.txt           # Original input transcript
├── 01_transcript.json              # Normalized transcript with segments
├── 02_claims.json                  # All extracted claims
├── 02_claims.reviewed.json         # Claims after review (if --review used)
├── 03_sources.json                 # Evidence sources retrieved
├── 04_snippets.json                # Relevant evidence snippets
├── 05_verdicts.json                # Verification results for each claim
├── 06_scorecard.md                 # Summary scorecard (0-100 score)
├── 07_08_review_outline_and_script.md  # YouTube review video script
├── run.json                        # Run metadata and configuration
└── run.log                         # Execution log with timing info
```

### Pipeline Stages

1. **Normalize** - Parse and structure transcript segments
2. **Extract** - LLM identifies claims using overlapping chunks
3. **Review** (optional) - Interactive claim editing
4. **Retrieve** - Web search with tiered source quality filtering
5. **Verify** - LLM evaluation with citations and rhetorical analysis
6. **Score** - Aggregate verdicts (0-100 scale)
7. **Write** - Generate video script and outline