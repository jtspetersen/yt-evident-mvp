# Plan: Stress-Test Dev Branch for MVP Efficiency

## Goal
Create a `dev/stress-test` branch to test the pipeline with real-world 30-60 min transcripts, tuning models and budgets for reliable output without crashes.

---

## Step 1: Create the dev branch
- Branch from current `sweet-sammet` HEAD: `git checkout -b dev/stress-test`

## Step 2: Upgrade models in `config.yaml`

Based on research for your AMD RX 7900 XTX (24GB VRAM / 64GB RAM):

| Stage | Current | New | Why |
|---|---|---|---|
| **Extract** | `qwen2.5:7b` | `qwen3:8b` | Matches qwen2.5:14b quality at same size/speed; better JSON compliance |
| **Verify** | `qwen3:30b` | `qwen3:30b` (keep) | Already optimal — MoE with only 3B active params, ~34 t/s, fits 21GB VRAM |
| **Write** | `qwen3:30b` | `gemma3:27b` | Top-ranked creative writing model at this size; richer script prose |

Temperature changes:
- Extract: `0.1` → `0.0` (tighten JSON reliability)
- Write: `0.3` → `0.5` (more creative output)

## Step 3: Raise budgets for longer transcripts

For 30-60 min videos (5K-10K words, denser claim coverage):

| Setting | Current | New | Rationale |
|---|---|---|---|
| `max_claims` | 15 | 25 | Longer videos have more checkable claims |
| `max_sources_per_claim` | 4 | 5 | More evidence per claim = better verdicts |
| `max_fetches_per_run` | 40 | 80 | 25 claims × 5 sources needs more fetch budget |
| `fetch_timeout_sec` | 20 | 25 | Slightly more patient with slow sites |
| `snippets_per_source` | 3 | 4 | More snippet coverage per source |
| `snippet_max_chars` | 800 | 1200 | Longer snippets = more context for verification |
| `second_pass_max_claims` | 8 | 12 | More re-verification capacity |
| `second_pass_extra_fetches` | 40 | 60 | More budget for second pass |

## Step 4: Add graceful error handling in `verify_claims.py`

Currently if JSON parsing fails after retry, the pipeline crashes. Add a fallback:
- If verification fails for a single claim after all retries, catch the exception and return a default UNCERTAIN verdict with `confidence=0.3` and a note explaining the failure.
- This prevents one bad claim from killing the entire run.

## Step 5: Add graceful error handling in `extract_claims.py`

Same principle — if claim extraction JSON parsing fails, log the error and return an empty claims list with a warning rather than crashing. The pipeline already handles 0 claims (early exit).

## Step 6: Pull the new models via Ollama

Run:
```
ollama pull qwen3:8b
ollama pull gemma3:27b
```
(qwen3:30b should already be available)

## Step 7: Verify with a test run

Run the pipeline with a real transcript to confirm:
- All models load and respond
- Budgets work at the new limits
- No crashes on longer content
- Output quality is acceptable

---

## Files Modified
1. `config.yaml` — model names, temperatures, budget values
2. `app/pipeline/verify_claims.py` — fallback UNCERTAIN verdict on failure
3. `app/pipeline/extract_claims.py` — graceful failure on parse error

## Files NOT Modified
- `app/main.py` — no changes needed, budgets flow from config
- `app/tools/ollama_client.py` — timeout/retry logic is already adequate (900s, 2 retries)
