# app/main.py
import os
import json
import yaml
import time
import argparse
import re
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from dotenv import load_dotenv
from tqdm import tqdm

from app.tools.logger import make_run_logger
from app.store.run_index import append_run_index
from app.store.creator_profiles import append_creator_profile_event
from app.tools.fetch import FETCH_STATS  # cache/network stats
from app.tools.review import review_claims_interactive

from app.pipeline.ingest import normalize_transcript, write_json
from app.pipeline.extract_claims import extract_claims
from app.pipeline.retrieve_evidence import retrieve_for_claims
from app.pipeline.consolidate_claims import consolidate_claims
from app.pipeline.verify_claims import verify_one, verify_group
from app.pipeline.scorecard import tally
from app.pipeline.write_outputs import write_text, write_outline_and_script


# ----------------------------
# Utilities
# ----------------------------

def should_log(level: str) -> bool:
    """Check if we should log at the given level based on EVIDENT_LOG_LEVEL env var."""
    levels = {"DEBUG": 0, "INFO": 1, "WARNING": 2, "ERROR": 3}
    current_level = os.environ.get("EVIDENT_LOG_LEVEL", "INFO")
    return levels.get(level, 1) >= levels.get(current_level, 1)


def load_config():
    """
    Load configuration from config.yaml and apply environment variable overrides.
    Configuration hierarchy (highest to lowest precedence):
    1. Environment variables (EVIDENT_*)
    2. .env file
    3. config.yaml
    """
    # Load .env file if it exists (does not override existing env vars)
    load_dotenv(override=False)

    # Load base config from YAML
    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Apply environment variable overrides
    cfg["ollama"]["base_url"] = os.getenv(
        "EVIDENT_OLLAMA_BASE_URL",
        cfg["ollama"]["base_url"]
    )
    cfg["ollama"]["model_extract"] = os.getenv(
        "EVIDENT_MODEL_EXTRACT",
        cfg["ollama"]["model_extract"]
    )
    cfg["ollama"]["model_verify"] = os.getenv(
        "EVIDENT_MODEL_VERIFY",
        cfg["ollama"]["model_verify"]
    )
    cfg["ollama"]["model_write"] = os.getenv(
        "EVIDENT_MODEL_WRITE",
        cfg["ollama"]["model_write"]
    )
    cfg["ollama"]["temperature_extract"] = float(os.getenv(
        "EVIDENT_TEMPERATURE_EXTRACT",
        cfg["ollama"].get("temperature_extract", 0.1)
    ))
    cfg["ollama"]["temperature_verify"] = float(os.getenv(
        "EVIDENT_TEMPERATURE_VERIFY",
        cfg["ollama"].get("temperature_verify", 0.0)
    ))
    cfg["ollama"]["temperature_write"] = float(os.getenv(
        "EVIDENT_TEMPERATURE_WRITE",
        cfg["ollama"].get("temperature_write", 0.5)
    ))
    cfg["searx"]["base_url"] = os.getenv(
        "EVIDENT_SEARXNG_BASE_URL",
        cfg["searx"]["base_url"]
    )
    cfg["budgets"]["max_claims"] = int(os.getenv(
        "EVIDENT_MAX_CLAIMS",
        cfg.get("budgets", {}).get("max_claims", 25)
    ))
    cfg["cache"]["url_cache_days"] = int(os.getenv(
        "EVIDENT_CACHE_TTL_DAYS",
        cfg.get("cache", {}).get("url_cache_days", 7)
    ))

    return cfg


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
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


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
    parser.add_argument("--url", type=str, default=None,
                        help="YouTube video URL (alternative to --infile). Fetches captions or transcribes locally.")
    parser.add_argument("--channel", type=str, default=None,
                        help="Channel/creator name for this run. If omitted, inferred from filename or YouTube metadata.")
    parser.add_argument("--review", action="store_true",
                        help="Interactive review: keep/drop/edit claims before retrieval+verification.")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress DEBUG/INFO output (show only warnings and errors).")
    parser.add_argument("--verbose", action="store_true",
                        help="Show all DEBUG output (overrides --quiet).")
    args = parser.parse_args()

    # Set logging level based on CLI flags
    if args.verbose:
        os.environ["EVIDENT_LOG_LEVEL"] = "DEBUG"
    elif args.quiet:
        os.environ["EVIDENT_LOG_LEVEL"] = "WARNING"
    else:
        os.environ["EVIDENT_LOG_LEVEL"] = "INFO"

    if args.url and args.infile:
        parser.error("--url and --infile are mutually exclusive. Provide one or the other.")

    cfg = load_config()

    # Resolve transcript input: YouTube URL or local file
    yt_meta = None
    if args.url:
        from app.tools.youtube import fetch_youtube_transcript
        print(f"Fetching transcript from YouTube: {args.url}")

        def _yt_progress(data):
            if not args.quiet:
                status = data.get("status", "")
                detail = data.get("detail", "")
                print(f"  [{status}] {detail}")

        yt_result = fetch_youtube_transcript(args.url, progress_callback=_yt_progress)
        raw_text = yt_result["raw_text"]
        yt_meta = yt_result["metadata"]

        # Save transcript to inbox/ for record-keeping
        safe_title = slugify(yt_meta.get("title") or yt_result["video_id"], max_len=80)
        os.makedirs("inbox", exist_ok=True)
        infile = os.path.join("inbox", f"{safe_title}.txt")
        with open(infile, "w", encoding="utf-8") as f:
            f.write(raw_text)
        print(f"Transcript saved to: {infile}")
    else:
        infile = pick_infile(args.infile)
        raw_text = read_file(infile)

    # Resolve channel name: --channel > YouTube metadata > filename inference
    if args.channel:
        channel = args.channel.strip()
    elif yt_meta and yt_meta.get("channel"):
        channel = yt_meta["channel"]
    else:
        channel = infer_channel_from_filename(infile) or "Unknown"

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
        "consolidate_claims",
        "review_claims_interactive",
        "retrieve_evidence",
        "check_claims",
        "scorecard",
        "fact_check_summary",
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
            "inferred": channel,
            "slug": channel_slug,
        },
        "transcript": {
            "base": transcript_base,
            "slug": transcript_slug,
        },
        "outdir": outdir,
        "youtube_url": args.url or None,
        "transcript_source": "youtube" if args.url else "file",
        "youtube_metadata": yt_meta,

        "config": {
            "ollama": {
                "base_url": cfg["ollama"]["base_url"],
                "model_extract": cfg["ollama"].get("model_extract"),
                "model_verify": cfg["ollama"].get("model_verify"),
                "model_write": cfg["ollama"].get("model_write"),
                "model_consolidate": cfg["ollama"].get("model_consolidate", cfg["ollama"].get("model_extract")),
                "model_verify_group": cfg["ollama"].get("model_verify_group", cfg["ollama"].get("model_verify")),
                "temperature_extract": cfg["ollama"].get("temperature_extract", 0.1),
                "temperature_verify": cfg["ollama"].get("temperature_verify", 0.0),
                "temperature_consolidate": cfg["ollama"].get("temperature_consolidate", 0.1),
            },
            "budgets": cfg.get("budgets", {}),
            "timezone": cfg.get("output", {}).get("timezone"),
            "deny_domains": (cfg.get("searx", {}).get("deny_domains") or []),
            "review_mode": bool(args.review),
        },

        "counts": {
            "claims_extracted": 0,
            "claims_consolidated": 0,
            "duplicates_removed": 0,
            "narrative_groups": 0,
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
        if yt_meta:
            transcript_json["video"]["title"] = yt_meta.get("title")
            transcript_json["video"]["url"] = yt_meta.get("url")
            transcript_json["video"]["channel"] = yt_meta.get("channel")
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
            temperature=cfg["ollama"].get("temperature_extract", 0.0),
            chunk_overlap=cfg["budgets"].get("extract_chunk_overlap", 8),
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

        # 2b) Consolidate claims (dedup + narrative grouping)
        log.log("Stage 2b: consolidate claims")
        s = stage_start(manifest, "consolidate_claims")
        groups = []
        if len(claims) >= 2:
            model_consolidate = cfg["ollama"].get("model_consolidate", cfg["ollama"]["model_extract"])
            temp_consolidate = cfg["ollama"].get("temperature_consolidate", 0.1)
            claims, groups = consolidate_claims(
                cfg["ollama"]["base_url"],
                model_consolidate,
                claims,
                transcript_json,
                temperature=temp_consolidate,
            )
            # Save consolidated artifact
            consolidated_data = {
                "claims": [c.model_dump() for c in claims],
                "groups": [g.model_dump() for g in groups],
            }
            with open(os.path.join(outdir, "02b_claims.consolidated.json"), "w", encoding="utf-8") as f:
                json.dump(consolidated_data, f, ensure_ascii=False, indent=2)
            add_artifact(manifest, "claims_consolidated", "02b_claims.consolidated.json")

            manifest["counts"]["claims_consolidated"] = len(claims)
            manifest["counts"]["duplicates_removed"] = manifest["counts"]["claims_extracted"] - len(claims)
            manifest["counts"]["narrative_groups"] = len(groups)
            manifest["counts"]["claims_kept"] = len(claims)
            log.log(f"Consolidation: {manifest['counts']['duplicates_removed']} duplicates removed, {len(groups)} narrative groups")
        else:
            log.log("Skipping consolidation (fewer than 2 claims)")
        stage_end(manifest, "consolidate_claims", s)
        write_run_manifest(outdir, manifest)
        write_artifacts_index(outdir, manifest)

        # 2c) Review mode
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

        # Post-review group cleanup: remove dropped claims from groups
        if groups:
            kept_ids = {c.claim_id for c in claims}
            cleaned_groups = []
            for g in groups:
                g.claim_ids = [cid for cid in g.claim_ids if cid in kept_ids]
                if len(g.claim_ids) >= 2:
                    cleaned_groups.append(g)
            if len(cleaned_groups) != len(groups):
                log.log(f"Groups after review cleanup: {len(cleaned_groups)} (was {len(groups)})")
            groups = cleaned_groups
            manifest["counts"]["narrative_groups"] = len(groups)

        # 3) Retrieve evidence
        log.log("Stage 3: retrieve evidence")
        s = stage_start(manifest, "retrieve_evidence")

        budgets = cfg["budgets"]
        deny_domains = (cfg.get("searx", {}).get("deny_domains") or [])

        # Generate search queries via LLM (multi-query strategy)
        generated_queries = None
        if budgets.get("enable_query_generation", True):
            from app.tools.query_gen import generate_queries_batch
            model_query = cfg["ollama"].get("model_query_gen", cfg["ollama"]["model_extract"])
            temp_query = float(cfg["ollama"].get("temperature_query_gen", 0.3))
            num_q = int(budgets.get("queries_per_claim", 3))
            workers_q = int(budgets.get("query_gen_workers", 3))

            log.log(f"Generating search queries ({num_q} per claim, {workers_q} workers)...")
            generated_queries = generate_queries_batch(
                cfg["ollama"]["base_url"], model_query, claims,
                num_queries=num_q, temperature=temp_query, max_workers=workers_q,
            )
            log.log(f"Generated search queries for {len(generated_queries)} claims")

            with open(os.path.join(outdir, "02d_queries.json"), "w", encoding="utf-8") as f:
                json.dump(generated_queries, f, ensure_ascii=False, indent=2)
            add_artifact(manifest, "generated_queries", "02d_queries.json")

        sources, snippets, evidence_by_claim, fetch_failures = retrieve_for_claims(
            cfg["searx"]["base_url"],
            claims,
            budgets,
            cfg["output"]["timezone"],
            snippets_per_source=budgets.get("snippets_per_source", 4),
            snippet_max_chars=budgets.get("snippet_max_chars", 1200),
            deny_domains=deny_domains,
            generated_queries=generated_queries,
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

        # 4) Check claims
        log.log("Stage 4: check claims")
        s = stage_start(manifest, "check_claims")

        verify_workers = int(cfg["budgets"].get("verify_workers", 3))

        def _verify_task(claim):
            """Run verify_one for a single claim. Executed in a worker thread."""
            t0v = time.time()
            v = verify_one(
                cfg["ollama"]["base_url"],
                cfg["ollama"]["model_verify"],
                claim,
                evidence_by_claim.get(claim.claim_id, {}),
                outdir,
                temperature=cfg["ollama"].get("temperature_verify", 0.0),
                transcript_json=transcript_json,
            )
            return claim.claim_id, v, round(time.time() - t0v, 4)

        # Submit all claims, collect results preserving original order
        results_by_id = {}   # claim_id -> (verdict, sec)
        verify_bar = tqdm(
            total=len(claims),
            desc="Verifying claims",
            unit=" claim",
            bar_format="{desc}: {n_fmt}/{total_fmt} claims | {elapsed}",
            leave=False,
        )

        with ThreadPoolExecutor(max_workers=verify_workers) as executor:
            futures = {executor.submit(_verify_task, c): c for c in claims}
            for fut in as_completed(futures):
                claim_id, v, sec = fut.result()
                results_by_id[claim_id] = (v, sec)
                verify_bar.update(1)
                log.log(f"Verified {len(results_by_id)}/{len(claims)} claim_id={claim_id} rating={v.rating} conf={v.confidence} sec={sec}")

        verify_bar.close()

        # Rebuild lists in original claim order
        verdicts = []
        per_claim_secs = []
        for c in claims:
            v, sec = results_by_id[c.claim_id]
            verdicts.append(v)
            per_claim_secs.append(sec)

        stage_end(manifest, "check_claims", s)

        with open(os.path.join(outdir, "05_verdicts.json"), "w", encoding="utf-8") as f:
            json.dump([v.model_dump() for v in verdicts], f, ensure_ascii=False, indent=2)

        add_artifact(manifest, "verdicts", "05_verdicts.json")
        manifest["counts"]["verdicts"] = len(verdicts)

        if per_claim_secs:
            manifest["timings"].setdefault("check_claims", {})
            manifest["timings"]["check_claims"]["per_claim_sec"] = {
                "min": round(min(per_claim_secs), 4),
                "max": round(max(per_claim_secs), 4),
                "avg": round(sum(per_claim_secs) / len(per_claim_secs), 4),
                "n": len(per_claim_secs),
            }

        # Group verification (narrative-level)
        group_verdicts = []
        if groups:
            model_verify_group = cfg["ollama"].get("model_verify_group", cfg["ollama"]["model_verify"])
            log.log(f"Verifying {len(groups)} narrative groups")
            for gi, g in enumerate(groups, 1):
                log.log(f"Group {gi}/{len(groups)}: {g.group_id} — {g.narrative_thesis[:60]}...")
                gv = verify_group(
                    cfg["ollama"]["base_url"],
                    model_verify_group,
                    g,
                    claims,
                    verdicts,
                    evidence_by_claim,
                    transcript_json=transcript_json,
                    temperature=cfg["ollama"].get("temperature_verify", 0.0),
                )
                group_verdicts.append(gv)
                log.log(f"Group {g.group_id}: {gv.narrative_rating} (confidence={gv.narrative_confidence})")

            with open(os.path.join(outdir, "05b_group_verdicts.json"), "w", encoding="utf-8") as f:
                json.dump([gv.model_dump() for gv in group_verdicts], f, ensure_ascii=False, indent=2)
            add_artifact(manifest, "group_verdicts", "05b_group_verdicts.json")

        write_run_manifest(outdir, manifest)
        write_artifacts_index(outdir, manifest)

        # 5) Scorecard
        log.log("Stage 5: scorecard")
        s = stage_start(manifest, "scorecard")
        counts, red_flags, tiers = tally(verdicts)
        stage_end(manifest, "scorecard", s)

        scorecard_md = f"""# Evident Scorecard

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
            "verdict_counts": counts,
            "tiers": tiers,
            "red_flags": red_flags,
        }

        write_run_manifest(outdir, manifest)
        write_artifacts_index(outdir, manifest)

        # 6) Fact-check summary
        log.log("Stage 6: fact-check summary")
        s = stage_start(manifest, "fact_check_summary")
        writer_md = write_outline_and_script(
            cfg["ollama"]["base_url"],
            cfg["ollama"]["model_write"],
            transcript_json,
            verdicts,
            scorecard_md,
            claims,
            manifest["channel"]["raw"],
            groups=groups,
            group_verdicts=group_verdicts,
        )
        stage_end(manifest, "fact_check_summary", s)

        write_text(os.path.join(outdir, "07_summary.md"), writer_md)
        add_artifact(manifest, "writer_md", "07_summary.md")

        manifest["status"] = "ok"

        # Run index
        append_run_index(
            run_id=manifest["run_id"],
            input_file=manifest["infile"],
            outdir=manifest["outdir"],
            verdict_counts=manifest["scorecard"]["verdict_counts"],
            duration_sec=time.time() - t0,
        )

        # Creator profile memory
        topics = extract_topics_lightweight(raw_text, max_topics=8)
        manifest["topics"] = topics
        append_creator_profile_event(
            channel=manifest["channel"]["raw"],
            run_id=manifest["run_id"],
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