# app/main.py
import os
import json
import yaml
import time
import argparse
import re
import traceback
from datetime import datetime

from app.tools.logger import make_run_logger
from app.store.run_index import append_run_index
from app.store.creator_profiles import append_creator_profile_event
from app.tools.fetch import FETCH_STATS  # cache/network stats
from app.tools.review import review_claims_interactive

from app.pipeline.ingest import normalize_transcript, write_json
from app.pipeline.extract_claims import extract_claims
from app.pipeline.retrieve_evidence import retrieve_for_claims
from app.pipeline.verify_claims import verify_one
from app.pipeline.scorecard import score
from app.pipeline.write_outputs import write_text, write_outline_and_script


# ----------------------------
# Utilities
# ----------------------------

def load_config():
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def pick_infile(cli_infile: str | None) -> str:
    if cli_infile:
        if os.path.exists(cli_infile):
            return cli_infile
        alt = os.path.join(os.getcwd(), cli_infile)
        if os.path.exists(alt):
            return alt
        raise FileNotFoundError(f"--infile not found: {cli_infile}")

    inbox_files = [f for f in os.listdir("inbox") if f.lower().endswith((".txt", ".md"))]
    if not inbox_files:
        raise FileNotFoundError("No transcripts found in inbox/. Add a .txt or .md file or pass --infile.")
    inbox_files.sort(key=lambda x: os.path.getmtime(os.path.join("inbox", x)), reverse=True)
    return os.path.join("inbox", inbox_files[0])


def infer_channel_from_filename(infile: str) -> str:
    """
    Heuristic: if filename is like 'Channel - Title.txt', take 'Channel'.
    Otherwise return 'Unknown'.
    """
    base = os.path.basename(infile)
    base = re.sub(r"\.(txt|md)$", "", base, flags=re.IGNORECASE)
    if " - " in base:
        return base.split(" - ", 1)[0].strip() or "Unknown"
    return "Unknown"


def slugify(s: str, max_len: int = 40) -> str:
    """
    Windows-safe slug:
    - lower
    - remove punctuation
    - spaces -> _
    - cap length
    """
    s = (s or "").strip().lower()
    if not s:
        return "unknown"
    s = re.sub(r"[^\w\s-]", "", s)   # remove punctuation
    s = re.sub(r"\s+", "_", s)       # spaces -> _
    s = re.sub(r"_+", "_", s)
    return s[:max_len] if len(s) > max_len else s


def extract_topics_lightweight(text: str, max_topics: int = 8) -> list:
    """
    Super-light topic guesser (no models). Just to start building signal.
    """
    t = (text or "").lower()
    keywords = [
        ("ai", ["ai", "artificial intelligence", "llm", "gpt", "model"]),
        ("economy", ["inflation", "unemployment", "gdp", "recession", "economy"]),
        ("health", ["cancer", "diet", "coffee", "vaccine", "covid", "nutrition", "health"]),
        ("politics", ["election", "president", "congress", "senate", "policy", "government"]),
        ("climate", ["climate", "carbon", "emissions", "warming"]),
        ("tech", ["software", "hardware", "chip", "open source", "github"]),
        ("science", ["study", "research", "meta-analysis", "trial", "paper", "journal"]),
        ("finance", ["stocks", "bond", "interest rate", "fed", "bitcoin", "crypto"]),
    ]
    hits = []
    for label, terms in keywords:
        if any(term in t for term in terms):
            hits.append(label)
    return hits[:max_topics]


def utc_now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def write_run_manifest(outdir: str, manifest: dict) -> None:
    path = os.path.join(outdir, "run.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def stage_start(manifest: dict, stage_key: str) -> float:
    manifest.setdefault("timings", {})
    manifest["timings"].setdefault(stage_key, {})
    manifest["timings"][stage_key]["started_utc"] = utc_now_iso()
    return time.perf_counter()


def stage_end(manifest: dict, stage_key: str, perf_start: float) -> float:
    dur = time.perf_counter() - perf_start
    manifest["timings"][stage_key]["finished_utc"] = utc_now_iso()
    manifest["timings"][stage_key]["duration_sec"] = round(dur, 4)
    return dur


def add_artifact(manifest: dict, key: str, filename: str) -> None:
    manifest.setdefault("artifacts", {})
    manifest.setdefault("outputs", [])
    manifest["artifacts"][key] = filename
    if filename not in manifest["outputs"]:
        manifest["outputs"].append(filename)


def format_seconds(sec: float | None) -> str:
    if sec is None:
        return "—"
    try:
        return f"{float(sec):.2f}s"
    except Exception:
        return "—"


def write_artifacts_index(outdir: str, manifest: dict) -> None:
    """
    Human-friendly index of run outputs.
    Written repeatedly during run so it's always available even mid-run.
    """
    artifacts = manifest.get("artifacts", {}) or {}
    timings = manifest.get("timings", {}) or {}

    def art(key: str) -> str:
        fn = artifacts.get(key)
        return fn if fn else "—"

    stage_lines = []
    stage_order = manifest.get("stage_order", []) or []
    for sk in stage_order:
        if sk in timings:
            d = timings.get(sk, {}).get("duration_sec")
            stage_lines.append(f"- `{sk}` — {format_seconds(d)}")
        else:
            stage_lines.append(f"- `{sk}` — (not run)")

    overall = manifest.get("scorecard", {}).get("overall")
    status = manifest.get("status")
    run_id = manifest.get("run_id")

    error_block = ""
    if status == "error" and manifest.get("error"):
        err = manifest["error"]
        error_block = (
            "\n## Error\n"
            f"- **Type:** `{err.get('type')}`\n"
            f"- **Message:** {err.get('message')}\n"
            f"- **See:** `run.log` for traceback\n"
        )

    md = f"""# Evident Run Artifacts

- **Run ID:** `{run_id}`
- **Status:** `{status}`
- **Overall score:** `{overall}`

## Quick links (files)
- **Manifest:** `{art('manifest')}`
- **Log:** `{art('log')}`
- **Raw transcript:** `{art('transcript_raw')}`
- **Normalized transcript:** `{art('transcript_normalized')}`
- **Claims:** `{art('claims')}`
- **Claims (reviewed):** `{art('claims_reviewed')}`
- **Sources:** `{art('sources')}`
- **Snippets:** `{art('snippets')}`
- **Evidence by claim:** `{art('evidence_by_claim')}`
- **Fetch failures:** `{art('fetch_failures')}`
- **Verdicts:** `{art('verdicts')}`
- **Scorecard:** `{art('scorecard_md')}`
- **Writer output:** `{art('writer_md')}`

## Stage timings
{chr(10).join(stage_lines)}
{error_block}
"""
    path = os.path.join(outdir, "artifacts_index.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(md)


# ----------------------------
# Main
# ----------------------------

def main():
    parser = argparse.ArgumentParser(description="Evident local fact-check MVP")
    parser.add_argument("--infile", type=str, default=None,
                        help="Path to transcript (.txt or .md). Defaults to newest file in inbox/.")
    parser.add_argument("--channel", type=str, default=None,
                        help="Channel/creator name for this run. If omitted, inferred from filename 'Channel - Title.txt'.")
    parser.add_argument("--review", action="store_true",
                        help="Interactive review: keep/drop/edit claims before retrieval+verification.")
    args = parser.parse_args()

    cfg = load_config()
    infile = pick_infile(args.infile)
    raw_text = read_file(infile)

    inferred_channel = infer_channel_from_filename(infile)
    channel = (args.channel or inferred_channel).strip() if (args.channel or infile) else "Unknown"

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    transcript_name = os.path.basename(infile)
    transcript_base = re.sub(r"\.(txt|md)$", "", transcript_name, flags=re.IGNORECASE)

    channel_slug = slugify(channel, max_len=40)
    transcript_slug = slugify(transcript_base, max_len=60)

    run_folder = f"{run_id}__{channel_slug}__{transcript_slug}"
    outdir = os.path.join("runs", run_folder)
    os.makedirs(outdir, exist_ok=True)

    log = make_run_logger(run_id, outdir=outdir)

    stage_order = [
        "normalize_transcript",
        "extract_claims",
        "review_claims_interactive",
        "retrieve_evidence",
        "verify_claims",
        "scorecard",
        "write_outline_and_script",
    ]

    t0 = time.time()
    manifest = {
        "manifest_version": "1.0",
        "status": "running",  # running | ok | error | early_exit

        "run_id": run_id,
        "started_utc": utc_now_iso(),
        "finished_utc": None,
        "duration_sec": None,

        "infile": infile,
        "transcript_filename": transcript_name,

        "channel": {
            "raw": channel,
            "inferred": inferred_channel,
            "slug": channel_slug,
        },
        "transcript": {
            "base": transcript_base,
            "slug": transcript_slug,
        },
        "outdir": outdir,

        "config": {
            "ollama": {
                "base_url": cfg["ollama"]["base_url"],
                "model_extract": cfg["ollama"].get("model_extract"),
                "model_verify": cfg["ollama"].get("model_verify"),
                "model_write": cfg["ollama"].get("model_write"),
                "temperature_extract": cfg["ollama"].get("temperature_extract", 0.1),
                "temperature_verify": cfg["ollama"].get("temperature_verify", 0.0),
            },
            "budgets": cfg.get("budgets", {}),
            "timezone": cfg.get("output", {}).get("timezone"),
            "deny_domains": (cfg.get("searx", {}).get("deny_domains") or []),
            "review_mode": bool(args.review),
        },

        "counts": {
            "claims_extracted": 0,
            "claims_kept": 0,
            "sources": 0,
            "snippets": 0,
            "fetch_failures": 0,
            "verdicts": 0,
        },

        "fetch_cache": {},
        "scorecard": {},
        "topics": [],

        "timings": {},
        "stage_order": stage_order,

        "error": None,
        "artifacts": {},
        "outputs": [],
    }

    # Always include these
    add_artifact(manifest, "manifest", "run.json")
    add_artifact(manifest, "log", "run.log")
    add_artifact(manifest, "artifacts_index", "artifacts_index.md")

    # Snapshot raw transcript
    raw_copy_name = "00_transcript.raw.txt"
    with open(os.path.join(outdir, raw_copy_name), "w", encoding="utf-8") as f:
        f.write(raw_text)
    add_artifact(manifest, "transcript_raw", raw_copy_name)

    # Persist immediately
    write_run_manifest(outdir, manifest)
    write_artifacts_index(outdir, manifest)

    try:
        log.log(f"RUN START run_id={run_id} infile={infile} channel={channel}")
        log.log(f"Outdir: {outdir}")

        # 1) Normalize transcript
        log.log("Stage 1: normalize transcript")
        s = stage_start(manifest, "normalize_transcript")
        transcript_json = normalize_transcript(raw_text)
        write_json(os.path.join(outdir, "01_transcript.normalized.json"), transcript_json)
        stage_end(manifest, "normalize_transcript", s)
        add_artifact(manifest, "transcript_normalized", "01_transcript.normalized.json")
        write_run_manifest(outdir, manifest)
        write_artifacts_index(outdir, manifest)

        # 2) Extract claims
        log.log("Stage 2: extract claims")
        s = stage_start(manifest, "extract_claims")
        claims = extract_claims(
            cfg["ollama"]["base_url"],
            cfg["ollama"]["model_extract"],
            transcript_json,
            max_claims=cfg["budgets"]["max_claims"],
            temperature=cfg["ollama"].get("temperature_extract", 0.1),
        )
        stage_end(manifest, "extract_claims", s)

        with open(os.path.join(outdir, "02_claims.json"), "w", encoding="utf-8") as f:
            json.dump([c.model_dump() for c in claims], f, ensure_ascii=False, indent=2)

        log.log(f"Claims extracted: {len(claims)}")
        manifest["counts"]["claims_extracted"] = len(claims)
        manifest["counts"]["claims_kept"] = len(claims)
        add_artifact(manifest, "claims", "02_claims.json")
        write_run_manifest(outdir, manifest)
        write_artifacts_index(outdir, manifest)

        # 2b) Review mode
        if args.review:
            log.log("Stage 2b: review claims (interactive)")
            s = stage_start(manifest, "review_claims_interactive")
            reviewed = review_claims_interactive(claims)
            stage_end(manifest, "review_claims_interactive", s)

            log.log(f"Claims kept after review: {len(reviewed)}")
            claims = reviewed

            with open(os.path.join(outdir, "02_claims.reviewed.json"), "w", encoding="utf-8") as f:
                json.dump([c.model_dump() for c in claims], f, ensure_ascii=False, indent=2)

            manifest["counts"]["claims_kept"] = len(claims)
            add_artifact(manifest, "claims_reviewed", "02_claims.reviewed.json")
            write_run_manifest(outdir, manifest)
            write_artifacts_index(outdir, manifest)

            if not claims:
                log.log("No claims kept. Ending run early.")
                manifest["status"] = "early_exit"
                manifest["topics"] = extract_topics_lightweight(raw_text, max_topics=8)
                write_run_manifest(outdir, manifest)
                write_artifacts_index(outdir, manifest)
                print(f"No claims kept. Outputs in: {outdir}")
                return

        # 3) Retrieve evidence
        log.log("Stage 3: retrieve evidence")
        s = stage_start(manifest, "retrieve_evidence")

        budgets = cfg["budgets"]
        deny_domains = (cfg.get("searx", {}).get("deny_domains") or [])

        sources, snippets, evidence_by_claim, fetch_failures = retrieve_for_claims(
            cfg["searx"]["base_url"],
            claims,
            budgets,
            cfg["output"]["timezone"],
            snippets_per_source=budgets.get("snippets_per_source", 4),
            snippet_max_chars=budgets.get("snippet_max_chars", 1200),
            deny_domains=deny_domains,
        )

        stage_end(manifest, "retrieve_evidence", s)

        with open(os.path.join(outdir, "03_sources.json"), "w", encoding="utf-8") as f:
            json.dump(sources, f, ensure_ascii=False, indent=2)
        with open(os.path.join(outdir, "03_snippets.json"), "w", encoding="utf-8") as f:
            json.dump(snippets, f, ensure_ascii=False, indent=2)
        with open(os.path.join(outdir, "04_evidence_by_claim.json"), "w", encoding="utf-8") as f:
            json.dump(evidence_by_claim, f, ensure_ascii=False, indent=2)
        with open(os.path.join(outdir, "04_fetch_failures.json"), "w", encoding="utf-8") as f:
            json.dump(fetch_failures, f, ensure_ascii=False, indent=2)

        add_artifact(manifest, "sources", "03_sources.json")
        add_artifact(manifest, "snippets", "03_snippets.json")
        add_artifact(manifest, "evidence_by_claim", "04_evidence_by_claim.json")
        add_artifact(manifest, "fetch_failures", "04_fetch_failures.json")

        hits = FETCH_STATS["cache_hit_ok"] + FETCH_STATS["cache_hit_fail"]
        misses = FETCH_STATS["cache_miss"]
        total = hits + misses
        hit_rate = (hits / total) if total else 0.0

        manifest["counts"]["sources"] = len(sources)
        manifest["counts"]["snippets"] = len(snippets)
        manifest["counts"]["fetch_failures"] = len(fetch_failures)
        manifest["fetch_cache"] = {
            "cache_hit_ok": FETCH_STATS["cache_hit_ok"],
            "cache_hit_fail": FETCH_STATS["cache_hit_fail"],
            "cache_miss": FETCH_STATS["cache_miss"],
            "net_ok": FETCH_STATS["net_ok"],
            "net_fail": FETCH_STATS["net_fail"],
            "hit_rate": round(hit_rate, 4),
        }

        write_run_manifest(outdir, manifest)
        write_artifacts_index(outdir, manifest)

        # 4) Verify claims
        log.log("Stage 4: verify claims")
        s = stage_start(manifest, "verify_claims")

        verdicts = []
        per_claim_secs = []
        for idx, c in enumerate(claims, start=1):
            bundle = evidence_by_claim.get(c.claim_id, {})
            t_claim = time.time()
            v = verify_one(
                cfg["ollama"]["base_url"],
                cfg["ollama"]["model_verify"],
                c,
                bundle,
                outdir,
                temperature=cfg["ollama"].get("temperature_verify", 0.0),
            )
            verdicts.append(v)
            sec = round(time.time() - t_claim, 4)
            per_claim_secs.append(sec)
            log.log(f"Verified {idx}/{len(claims)} claim_id={c.claim_id} rating={v.rating} conf={v.confidence} sec={round(sec,2)}")

        stage_end(manifest, "verify_claims", s)

        with open(os.path.join(outdir, "05_verdicts.json"), "w", encoding="utf-8") as f:
            json.dump([v.model_dump() for v in verdicts], f, ensure_ascii=False, indent=2)

        add_artifact(manifest, "verdicts", "05_verdicts.json")
        manifest["counts"]["verdicts"] = len(verdicts)

        if per_claim_secs:
            manifest["timings"].setdefault("verify_claims", {})
            manifest["timings"]["verify_claims"]["per_claim_sec"] = {
                "min": round(min(per_claim_secs), 4),
                "max": round(max(per_claim_secs), 4),
                "avg": round(sum(per_claim_secs) / len(per_claim_secs), 4),
                "n": len(per_claim_secs),
            }

        write_run_manifest(outdir, manifest)
        write_artifacts_index(outdir, manifest)

        # 5) Scorecard
        log.log("Stage 5: scorecard")
        s = stage_start(manifest, "scorecard")
        overall, counts, red_flags, tiers = score(verdicts)
        stage_end(manifest, "scorecard", s)

        scorecard_md = f"""# Evident Scorecard

**Overall score:** {overall}/100

## Verdict counts
{json.dumps(counts, indent=2)}

## Source tiers used
{json.dumps(tiers, indent=2)}

## Red flags detected
{json.dumps(red_flags, indent=2)}
"""
        write_text(os.path.join(outdir, "06_scorecard.md"), scorecard_md)
        add_artifact(manifest, "scorecard_md", "06_scorecard.md")

        manifest["scorecard"] = {
            "overall": int(overall),
            "verdict_counts": counts,
            "tiers": tiers,
            "red_flags": red_flags,
        }

        write_run_manifest(outdir, manifest)
        write_artifacts_index(outdir, manifest)

        # 6) Writer
        log.log("Stage 6: write outline + script")
        s = stage_start(manifest, "write_outline_and_script")
        writer_md = write_outline_and_script(
            cfg["ollama"]["base_url"],
            cfg["ollama"]["model_write"],
            transcript_json,
            verdicts,
            scorecard_md,
        )
        stage_end(manifest, "write_outline_and_script", s)

        write_text(os.path.join(outdir, "07_08_review_outline_and_script.md"), writer_md)
        add_artifact(manifest, "writer_md", "07_08_review_outline_and_script.md")

        manifest["status"] = "ok"

        # Run index
        append_run_index(
            run_id=manifest["run_id"],
            input_file=manifest["infile"],
            outdir=manifest["outdir"],
            overall_score=int(manifest["scorecard"]["overall"]),
            verdict_counts=manifest["scorecard"]["verdict_counts"],
            duration_sec=time.time() - t0,
        )

        # Creator profile memory
        topics = extract_topics_lightweight(raw_text, max_topics=8)
        manifest["topics"] = topics
        append_creator_profile_event(
            channel=manifest["channel"]["raw"],
            run_id=manifest["run_id"],
            overall_score=int(manifest["scorecard"]["overall"]),
            verdict_counts=manifest["scorecard"]["verdict_counts"],
            red_flags=manifest["scorecard"].get("red_flags", []) if isinstance(manifest["scorecard"].get("red_flags"), list) else [],
            topics=topics,
            input_file=manifest["infile"],
            outdir=manifest["outdir"],
        )

        write_run_manifest(outdir, manifest)
        write_artifacts_index(outdir, manifest)

    except Exception as e:
        manifest["status"] = "error"
        manifest["error"] = {
            "type": type(e).__name__,
            "message": str(e),
            "traceback": traceback.format_exc(),
        }
        log.log(f"ERROR: {type(e).__name__}: {e}")
        log.log(manifest["error"]["traceback"])
        write_run_manifest(outdir, manifest)
        write_artifacts_index(outdir, manifest)
        raise

    finally:
        duration = time.time() - t0
        manifest["finished_utc"] = utc_now_iso()
        manifest["duration_sec"] = round(duration, 3)
        write_run_manifest(outdir, manifest)
        write_artifacts_index(outdir, manifest)
        log.log(f"RUN END status={manifest.get('status')} duration_sec={round(duration,2)} outdir={outdir}")

    print(f"Done. Outputs in: {outdir}")


if __name__ == "__main__":
    main()