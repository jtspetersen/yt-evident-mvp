# Evident Video Fact Checker

A local fact-checking pipeline for analyzing video transcripts and verifying claims using evidence-based research.

## Features

- **Transcript ingestion** - Handles fragmented YouTube transcripts with speaker detection
- **Claim extraction** - Overlapping chunks prevent missing claims at segment boundaries
- **Evidence retrieval** - 6-tier quality system prioritizes scholarly sources over forums/blogs
- **Claim verification** - LLM reasoning with citations, confidence scoring, and rhetorical analysis
- **Report generation** - Detailed verdicts with verdict count summaries, and video script outline

## Quick Start

### Option 1: Native Setup (Recommended for Windows with GPU)

```bash
# 1. Install dependencies
pip install -r Requirements.txt

# 2. Ensure services are running:
#    - Ollama (with GPU): ollama serve
#    - SearXNG: docker compose -f docker/docker-compose.yml up -d searxng redis

# 3. Run the pipeline
./run.sh --infile "inbox/transcript.txt" --channel "Channel Name"
```

### Option 2: Full Docker Setup

```bash
# 1. Run interactive setup wizard
python setup.py

# 2. Start services
docker compose -f docker/docker-compose.yml up -d

# 3. Run pipeline (in Docker)
docker compose -f docker/docker-compose.yml run --rm app python -m app.main --infile inbox/transcript.txt
```

### Option 3: Web UI

A browser-based interface for uploading transcripts, monitoring progress in real time, reviewing claims, and viewing reports.

```bash
# Start the web server
python -m app.web.server

# Or via Make
make web
```

Then open **http://localhost:8000** in your browser.

**Web UI features:**
- Upload transcripts via drag-and-drop or file picker
- Real-time progress dashboard with per-stage progress bars (extract, retrieve, verify)
- Live counters for claims, sources, snippets, and failures
- Optional claim review step — edit or drop claims before verification
- Rendered report with verdict summary badges and artifact downloads
- Past runs history

The web UI uses the same pipeline as the CLI. No additional services are required beyond Ollama and SearXNG.

## Project Structure

```
evident-video-fact-checker/
├── app/                    # Application code
│   ├── main.py             # CLI entry point
│   ├── pipeline/           # Processing stages
│   ├── schemas/            # Pydantic models
│   ├── store/              # Store modules
│   ├── tools/              # Utilities (fetch, parse, ollama)
│   └── web/                # Web UI (FastAPI + HTMX)
│       ├── server.py       # Routes and SSE endpoint
│       ├── runner.py       # Background pipeline runner
│       ├── templates/      # Jinja2 HTML templates
│       └── static/         # Vendored CSS/JS (Pico.css, HTMX)
├── docker/                 # Docker configuration
│   ├── docker-compose.yml
│   ├── docker-compose.gpu.yml      # NVIDIA GPU override
│   ├── docker-compose.amd.yml      # AMD ROCm override
│   └── Dockerfile
├── inbox/                  # Input transcripts
├── runs/                   # Output directories (timestamped)
├── cache/                  # URL cache (gitignored)
├── store/                  # Persistent storage (gitignored)
├── searxng/                # SearXNG configuration
├── config.yaml             # Application config
├── .env                    # Environment variables
├── run.sh                  # Run script
├── setup.py                # Interactive setup wizard
└── Makefile                # Make commands
```

## Usage

### Command-Line Flags

| Flag | Description | Default |
|------|-------------|---------|
| `--infile <path>` | Path to transcript file | Newest in `inbox/` |
| `--channel <name>` | Channel/creator name | Inferred from filename |
| `--review` | Interactive claim review mode | Disabled |
| `--verbose` | Show DEBUG output | Disabled |
| `--quiet` | Errors/warnings only | Disabled |

### Examples

```bash
# Native execution (recommended for Windows with GPU)
./run.sh --infile "inbox/transcript.txt" --channel "Channel Name"

# With interactive review
./run.sh --infile "inbox/transcript.txt" --review --verbose

# Direct Python
python -m app.main --infile "inbox/transcript.txt" --channel "Channel Name"
```

## Configuration

### Environment Variables (.env)

```bash
# Service URLs
EVIDENT_OLLAMA_BASE_URL=http://localhost:11434
EVIDENT_SEARXNG_BASE_URL=http://localhost:8080

# Model selections
EVIDENT_MODEL_EXTRACT=qwen3:8b
EVIDENT_MODEL_VERIFY=qwen3:30b
EVIDENT_MODEL_WRITE=gemma3:27b
```

### config.yaml

```yaml
ollama:
  model_extract: "qwen3:8b"      # Claim extraction
  model_verify: "qwen3:30b"      # Verification
  model_write: "gemma3:27b"      # Script writing

budgets:
  max_claims: 25
  max_sources_per_claim: 5
  max_fetches_per_run: 80
```

### Verdict Ratings

Each claim receives one of six ratings:

| Rating | Meaning |
|--------|---------|
| VERIFIED | Confirmed by strong evidence |
| LIKELY TRUE | Supported but not fully confirmed |
| INSUFFICIENT EVIDENCE | Not enough quality sources found |
| CONFLICTING EVIDENCE | Credible sources disagree |
| LIKELY FALSE | Evidence suggests the claim is wrong |
| FALSE | Clearly contradicted by strong evidence |

### Source Quality Tiers

The pipeline uses a 6-tier source quality system:

| Tier | Description | Examples |
|------|-------------|----------|
| 1 | Top scholarly journals | Nature, Science, NEJM |
| 2 | Academic institutions | .edu, .ac.uk |
| 3 | Government/International orgs | .gov, WHO, UN |
| 4 | Research organizations | Pew Research, Brookings |
| 5 | Established news agencies | Reuters, AP, BBC |
| 6 | Everything else | - |

## Output

Each run creates a timestamped directory:

```
runs/YYYYMMDD_HHMMSS__channel__video_title/
├── 00_transcript.raw.txt           # Original input
├── 01_transcript.json              # Normalized segments
├── 02_claims.json                  # Extracted claims
├── 03_sources.json                 # Retrieved evidence
├── 04_snippets.json                # Evidence snippets
├── 05_verdicts.json                # Verification results
├── 06_scorecard.md                 # Verdict counts and source tiers
├── 07_summary.md                      # Video script
├── run.json                        # Run metadata
└── run.log                         # Execution log
```

## Requirements

- **Python 3.11+**
- **Ollama** - Local LLM server (GPU recommended)
- **SearXNG** - Metasearch engine for evidence retrieval
- **Redis** - For SearXNG caching

### Hardware Recommendations

| VRAM | RAM | Recommended Models |
|------|-----|-------------------|
| 24GB+ | 32GB+ | qwen3:30b, gemma3:27b |
| 12-16GB | 32GB+ | qwen3:14b, llama3:8b |
| 8GB | 16GB+ | qwen3:8b, llama3:8b |
| None | 32GB+ | qwen3:8b (CPU mode) |

## Make Commands

```bash
make help              # Show all commands
make setup             # Run setup wizard
make web               # Start web UI at http://localhost:8000
make runvid ARGS='...' # Run natively (recommended)
make start             # Start Docker services
make stop              # Stop Docker services
make status            # Show service status
make logs              # Tail all logs
make models            # List Ollama models
```

## Documentation

- [DOCKER.md](DOCKER.md) - Detailed Docker setup guide
- [MIGRATION.md](MIGRATION.md) - Migration from legacy setup
- [CLAUDE.md](CLAUDE.md) - Project context for AI assistants
