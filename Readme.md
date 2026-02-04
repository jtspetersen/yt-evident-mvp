# YT Evident - Fact Checker MVP

A local fact-checking pipeline for analyzing video transcripts and verifying claims using evidence-based research.

## Features

- Transcript ingestion and normalization
- Automated claim extraction
- Evidence retrieval from web sources
- Claim verification using LLM reasoning
- Scoring and report generation

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
└── run.sh             # Run script
```

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure `config.yaml`:**
   - Set your Ollama base URL
   - Configure search engine (SearX)
   - Adjust budgets and timings

3. **Run the fact-checker:**
   ```bash
   ./run.sh
   # or
   python -m app.main --infile inbox/your-transcript.txt
   ```

## Requirements

- Python 3.11+
- Ollama (for LLM verification)
- SearX instance (for evidence retrieval)

## Output

Each run creates a timestamped directory in `runs/` containing:
- Claims extracted
- Evidence retrieved
- Verdicts for each claim
- Overall scorecard
- Review script outline