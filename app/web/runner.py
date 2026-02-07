# app/web/runner.py
"""
Background pipeline runner for the web UI.

Wraps the existing pipeline functions (extract_claims, retrieve_for_claims,
verify_one, etc.) in a class that runs in a background thread, emits events
to a queue for SSE streaming, and can pause at the review stage waiting for
web input via threading.Event.
"""
import os
import json
import time
import queue
import re
import threading
import traceback
from datetime import datetime, timezone

from app.tools.logger import make_run_logger
from app.store.run_index import append_run_index
from app.store.creator_profiles import append_creator_profile_event
from app.tools.fetch import FETCH_STATS

from app.pipeline.ingest import normalize_transcript, write_json
from app.pipeline.extract_claims import extract_claims
from app.pipeline.retrieve_evidence import retrieve_for_claims
from app.pipeline.consolidate_claims import consolidate_claims
from app.pipeline.verify_claims import verify_one, verify_group
from app.pipeline.scorecard import tally
from app.pipeline.write_outputs import write_text, write_outline_and_script


# ---------------------------------------------------------------------------
# Helpers (from main.py)
# ---------------------------------------------------------------------------

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _slugify(s: str, max_len: int = 40) -> str:
    s = (s or "").strip().lower()
    if not s:
        return "unknown"
    s = re.sub(r"[^\w\s-]", "", s)
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s[:max_len] if len(s) > max_len else s


def _extract_topics_lightweight(text: str, max_topics: int = 8) -> list:
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


def _infer_channel_from_filename(infile: str) -> str:
    base = os.path.basename(infile)
    base = re.sub(r"\.(txt|md)$", "", base, flags=re.IGNORECASE)
    if " - " in base:
        return base.split(" - ", 1)[0].strip() or "Unknown"
    return "Unknown"


def _format_seconds(sec) -> str:
    if sec is None:
        return "—"
    try:
        return f"{float(sec):.2f}s"
    except Exception:
        return "—"


# ---------------------------------------------------------------------------
# Pipeline Runner
# ---------------------------------------------------------------------------

class PipelineCancelled(Exception):
    """Raised when a pipeline run is stopped by the user."""
    pass


# Global registry of active runners (run_id -> PipelineRunner)
_RUNNERS = {}
_RUNNERS_LOCK = threading.Lock()


def get_runner(run_id: str):
    with _RUNNERS_LOCK:
        return _RUNNERS.get(run_id)


def list_runners():
    with _RUNNERS_LOCK:
        return {rid: r.status for rid, r in _RUNNERS.items()}


class PipelineRunner:
    """
    Runs the full fact-check pipeline in a background thread.

    Emits events to self.events (queue.Queue) for SSE streaming.
    Pauses at the review stage if review_enabled, waiting for
    submit_review() to be called.
    """

    STAGE_ORDER = [
        "fetch_transcript",
        "normalize_transcript",
        "extract_claims",
        "consolidate_claims",
        "review_claims",
        "retrieve_evidence",
        "check_claims",
        "scorecard",
        "fact_check_summary",
    ]

    def __init__(self, cfg: dict, infile: str, raw_text: str,
                 channel: str = None, review_enabled: bool = False,
                 youtube_url: str = None):
        self.cfg = cfg
        self.infile = infile           # Can be None when youtube_url is set
        self.raw_text = raw_text       # Can be None when youtube_url is set
        self.review_enabled = review_enabled
        self.youtube_url = youtube_url

        if infile:
            inferred_channel = _infer_channel_from_filename(infile)
        else:
            inferred_channel = "Unknown"
        self.channel = (channel or inferred_channel).strip() or "Unknown"

        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        if infile:
            transcript_name = os.path.basename(infile)
            transcript_base = re.sub(r"\.(txt|md)$", "", transcript_name, flags=re.IGNORECASE)
        else:
            transcript_name = "youtube_pending"
            transcript_base = "youtube_pending"
        channel_slug = _slugify(self.channel, max_len=40)
        transcript_slug = _slugify(transcript_base, max_len=60)

        run_folder = f"{self.run_id}__{channel_slug}__{transcript_slug}"
        self.outdir = os.path.join("runs", run_folder)
        os.makedirs(self.outdir, exist_ok=True)

        # SSE event queue
        self.events = queue.Queue()

        # Pipeline state
        self.status = "pending"  # pending → running → review → running → done | error
        self.current_stage = None
        self.claims = []
        self.groups = []
        self.group_verdicts = []
        self.verdicts = []
        self.manifest = {}
        self.report_md = None
        self.error_info = None

        # Review synchronization
        self.review_result = None
        self._review_event = threading.Event()

        # Stop / cancel
        self._stop_event = threading.Event()

        # Thread
        self._thread = None

        # Register globally
        with _RUNNERS_LOCK:
            _RUNNERS[self.run_id] = self

    def start(self):
        """Launch pipeline in background thread."""
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit_review(self, decisions: list):
        """
        Called by web POST to submit review decisions and unblock pipeline.

        decisions: list of dicts with keys:
            - claim_id: str
            - action: "keep" | "drop" | "edit"
            - claim_text: str (optional, for edits)
            - claim_type: str (optional, for edits)
            - quote_from_transcript: str (optional, for edits)
        """
        self.review_result = decisions
        self._review_event.set()

    def stop(self):
        """Signal the pipeline to stop at next checkpoint."""
        self._stop_event.set()
        # Also unblock review wait if stuck there
        self._review_event.set()

    def _check_stop(self):
        """Raise PipelineCancelled if stop was requested."""
        if self._stop_event.is_set():
            raise PipelineCancelled()

    def emit(self, event_type: str, data: dict):
        """Push event to SSE queue."""
        self.events.put({
            "event": event_type,
            "data": json.dumps(data, ensure_ascii=False),
        })

    # ------------------------------------------------------------------
    # Manifest helpers (mirrors main.py)
    # ------------------------------------------------------------------

    def _add_artifact(self, key: str, filename: str):
        self.manifest.setdefault("artifacts", {})
        self.manifest.setdefault("outputs", [])
        self.manifest["artifacts"][key] = filename
        if filename not in self.manifest["outputs"]:
            self.manifest["outputs"].append(filename)

    def _stage_start(self, stage_key: str) -> float:
        self.manifest.setdefault("timings", {})
        self.manifest["timings"].setdefault(stage_key, {})
        self.manifest["timings"][stage_key]["started_utc"] = _utc_now_iso()
        self.current_stage = stage_key
        self.emit("stage", {"name": stage_key, "status": "started"})
        return time.perf_counter()

    def _stage_end(self, stage_key: str, perf_start: float) -> float:
        dur = time.perf_counter() - perf_start
        self.manifest["timings"][stage_key]["finished_utc"] = _utc_now_iso()
        self.manifest["timings"][stage_key]["duration_sec"] = round(dur, 4)
        self.emit("stage", {"name": stage_key, "status": "completed", "duration_sec": round(dur, 2)})
        return dur

    def _save_manifest(self):
        path = os.path.join(self.outdir, "run.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.manifest, f, ensure_ascii=False, indent=2)

    def _log_and_emit(self, log, msg: str):
        log.log(msg)
        self.emit("log", {"message": msg})

    # ------------------------------------------------------------------
    # Main pipeline execution
    # ------------------------------------------------------------------

    def _run(self):
        """Execute full pipeline, pausing at review if enabled."""
        cfg = self.cfg
        t0 = time.time()
        log = make_run_logger(self.run_id, outdir=self.outdir)

        self.status = "running"
        self.emit("status", {"status": "running"})

        transcript_name = os.path.basename(self.infile)
        transcript_base = re.sub(r"\.(txt|md)$", "", transcript_name, flags=re.IGNORECASE)
        channel_slug = _slugify(self.channel, max_len=40)
        transcript_slug = _slugify(transcript_base, max_len=60)

        self.manifest = {
            "manifest_version": "1.0",
            "status": "running",
            "run_id": self.run_id,
            "started_utc": _utc_now_iso(),
            "finished_utc": None,
            "duration_sec": None,
            "infile": self.infile,
            "transcript_filename": transcript_name,
            "channel": {
                "raw": self.channel,
                "inferred": _infer_channel_from_filename(self.infile),
                "slug": channel_slug,
            },
            "transcript": {
                "base": transcript_base,
                "slug": transcript_slug,
            },
            "outdir": self.outdir,
            "youtube_url": self.youtube_url,
            "transcript_source": "youtube" if self.youtube_url else "file",
            "youtube_metadata": None,
            "config": {
                "ollama": {
                    "base_url": cfg["ollama"]["base_url"],
                    "model_extract": cfg["ollama"].get("model_extract"),
                    "model_verify": cfg["ollama"].get("model_verify"),
                    "model_write": cfg["ollama"].get("model_write"),
                    "model_consolidate": cfg["ollama"].get("model_consolidate", cfg["ollama"].get("model_extract")),
                    "model_verify_group": cfg["ollama"].get("model_verify_group", cfg["ollama"].get("model_verify")),
                    "model_query_gen": cfg["ollama"].get("model_query_gen", cfg["ollama"].get("model_extract")),
                    "temperature_extract": cfg["ollama"].get("temperature_extract", 0.1),
                    "temperature_verify": cfg["ollama"].get("temperature_verify", 0.0),
                    "temperature_consolidate": cfg["ollama"].get("temperature_consolidate", 0.1),
                },
                "budgets": cfg.get("budgets", {}),
                "timezone": cfg.get("output", {}).get("timezone"),
                "deny_domains": (cfg.get("searx", {}).get("deny_domains") or []),
                "review_mode": self.review_enabled,
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
            "stage_order": self.STAGE_ORDER,
            "error": None,
            "artifacts": {},
            "outputs": [],
        }

        self._add_artifact("manifest", "run.json")
        self._add_artifact("log", "run.log")

        # Save raw transcript (if available; YouTube mode saves after fetch)
        if self.raw_text:
            raw_copy_name = "00_transcript.raw.txt"
            with open(os.path.join(self.outdir, raw_copy_name), "w", encoding="utf-8") as f:
                f.write(self.raw_text)
            self._add_artifact("transcript_raw", raw_copy_name)
        self._save_manifest()

        try:
            self._log_and_emit(log, f"RUN START run_id={self.run_id} channel={self.channel}")

            # ----- Stage 0 (conditional): Fetch YouTube transcript -----
            if self.youtube_url:
                self._check_stop()
                self._log_and_emit(log, f"Fetching YouTube transcript from {self.youtube_url}")
                s = self._stage_start("fetch_transcript")

                from app.tools.youtube import fetch_youtube_transcript

                def _yt_progress(data):
                    self.emit("youtube_progress", data)
                    status = data.get("status", "")
                    detail = data.get("detail", "")
                    if status in ("trying_captions", "captions_found", "no_captions",
                                  "downloading_audio", "transcribing", "done"):
                        self._log_and_emit(log, f"YouTube: {status} — {detail}")

                yt_result = fetch_youtube_transcript(
                    self.youtube_url, progress_callback=_yt_progress,
                )
                self.raw_text = yt_result["raw_text"]
                meta = yt_result["metadata"]

                # Auto-set channel from YouTube metadata
                if meta.get("channel") and self.channel in ("Unknown", ""):
                    self.channel = meta["channel"]

                # Save transcript to inbox/ and update self.infile
                safe_title = _slugify(meta.get("title") or yt_result["video_id"], max_len=80)
                self.infile = os.path.join("inbox", f"{safe_title}.txt")
                os.makedirs("inbox", exist_ok=True)
                with open(self.infile, "w", encoding="utf-8") as f:
                    f.write(self.raw_text)

                # Save raw transcript to run output
                raw_copy_name = "00_transcript.raw.txt"
                with open(os.path.join(self.outdir, raw_copy_name), "w", encoding="utf-8") as f:
                    f.write(self.raw_text)
                self._add_artifact("transcript_raw", raw_copy_name)

                # Update manifest with YouTube info
                self.manifest["youtube_metadata"] = meta
                self.manifest["transcript_source"] = yt_result.get("source", "unknown")
                self.manifest["channel"]["raw"] = self.channel
                self.manifest["infile"] = self.infile
                self.manifest["transcript_filename"] = os.path.basename(self.infile)

                self._stage_end("fetch_transcript", s)
                self._log_and_emit(log, f"YouTube transcript fetched ({yt_result['source']}): {len(self.raw_text)} chars")
                self._save_manifest()
                self._check_stop()
            else:
                # Mark fetch_transcript as skipped for file-upload runs
                self.manifest.setdefault("timings", {})
                self.manifest["timings"]["fetch_transcript"] = {"skipped": True}

            # ----- Stage 1: Normalize transcript -----
            self._check_stop()
            self._log_and_emit(log, "Stage 1: normalize transcript")
            s = self._stage_start("normalize_transcript")
            transcript_json = normalize_transcript(self.raw_text)
            # Populate video metadata from YouTube if available
            if self.manifest.get("youtube_metadata"):
                yt_meta = self.manifest["youtube_metadata"]
                transcript_json["video"]["title"] = yt_meta.get("title")
                transcript_json["video"]["url"] = yt_meta.get("url")
                transcript_json["video"]["channel"] = yt_meta.get("channel")
            write_json(os.path.join(self.outdir, "01_transcript.normalized.json"), transcript_json)
            self._stage_end("normalize_transcript", s)
            self._add_artifact("transcript_normalized", "01_transcript.normalized.json")
            self._save_manifest()

            # ----- Stage 2: Extract claims -----
            self._check_stop()
            self._log_and_emit(log, "Stage 2: extract claims")
            s = self._stage_start("extract_claims")

            def _extract_progress(data):
                self.emit("extract_progress", data)
                status = data.get("status", "")
                chunk = data.get("chunk", 0)
                total = data.get("total_chunks", 1)
                claims_so_far = data.get("claims_so_far", 0)
                if status == "extracting":
                    self._log_and_emit(log, f"Extracting chunk {chunk}/{total}... ({claims_so_far} claims so far)")
                elif status == "chunk_done":
                    self._log_and_emit(log, f"Chunk {chunk}/{total} done — {data.get('chunk_claims', 0)} new claims ({claims_so_far} total)")

            claims = extract_claims(
                cfg["ollama"]["base_url"],
                cfg["ollama"]["model_extract"],
                transcript_json,
                max_claims=cfg["budgets"]["max_claims"],
                temperature=cfg["ollama"].get("temperature_extract", 0.0),
                chunk_overlap=cfg["budgets"].get("extract_chunk_overlap", 8),
                progress_callback=_extract_progress,
            )

            # Emit counts BEFORE stage_end so UI counters update
            # while the detail panel is still visible
            self._log_and_emit(log, f"Claims extracted: {len(claims)}")
            self.claims = claims
            self.manifest["counts"]["claims_extracted"] = len(claims)
            self.manifest["counts"]["claims_kept"] = len(claims)
            self.emit("progress", {
                "stage": "extract_claims",
                "claims_extracted": len(claims),
            })

            self._stage_end("extract_claims", s)

            with open(os.path.join(self.outdir, "02_claims.json"), "w", encoding="utf-8") as f:
                json.dump([c.model_dump() for c in claims], f, ensure_ascii=False, indent=2)
            self._add_artifact("claims", "02_claims.json")
            self._save_manifest()

            # ----- Stage 2b: Consolidate claims -----
            self._check_stop()
            self._log_and_emit(log, "Stage 2b: consolidate claims")
            s = self._stage_start("consolidate_claims")
            groups = []
            if len(claims) >= 2:
                model_consolidate = cfg["ollama"].get("model_consolidate", cfg["ollama"]["model_extract"])
                temp_consolidate = cfg["ollama"].get("temperature_consolidate", 0.1)

                def _consolidate_progress(data):
                    self.emit("consolidate_progress", data)

                claims, groups = consolidate_claims(
                    cfg["ollama"]["base_url"],
                    model_consolidate,
                    claims,
                    transcript_json,
                    temperature=temp_consolidate,
                    progress_callback=_consolidate_progress,
                )

                consolidated_data = {
                    "claims": [c.model_dump() for c in claims],
                    "groups": [g.model_dump() for g in groups],
                }
                with open(os.path.join(self.outdir, "02b_claims.consolidated.json"), "w", encoding="utf-8") as f:
                    json.dump(consolidated_data, f, ensure_ascii=False, indent=2)
                self._add_artifact("claims_consolidated", "02b_claims.consolidated.json")

                dupes = self.manifest["counts"]["claims_extracted"] - len(claims)
                self.manifest["counts"]["claims_consolidated"] = len(claims)
                self.manifest["counts"]["duplicates_removed"] = dupes
                self.manifest["counts"]["narrative_groups"] = len(groups)
                self.manifest["counts"]["claims_kept"] = len(claims)
                self.claims = claims
                self.groups = groups
                self._log_and_emit(log, f"Consolidation: {dupes} duplicates removed, {len(groups)} narrative groups")
            else:
                self._log_and_emit(log, "Skipping consolidation (fewer than 2 claims)")
            self._stage_end("consolidate_claims", s)
            self._save_manifest()

            # ----- Stage 2c: Review (pause if enabled) -----
            if self.review_enabled:
                self._log_and_emit(log, "Stage 2b: review claims (waiting for web input)")
                s = self._stage_start("review_claims")
                self.status = "review"

                # Send claims to frontend
                self.emit("review_ready", {
                    "claims": [c.model_dump() for c in claims],
                })

                # Block until submit_review() is called
                self._review_event.wait()

                # Apply review decisions
                if self.review_result is not None:
                    claims = self._apply_review(claims, self.review_result)

                self._stage_end("review_claims", s)
                self.status = "running"
                self.emit("status", {"status": "running"})

                self._log_and_emit(log, f"Claims kept after review: {len(claims)}")
                self.claims = claims

                with open(os.path.join(self.outdir, "02_claims.reviewed.json"), "w", encoding="utf-8") as f:
                    json.dump([c.model_dump() for c in claims], f, ensure_ascii=False, indent=2)

                self.manifest["counts"]["claims_kept"] = len(claims)
                self._add_artifact("claims_reviewed", "02_claims.reviewed.json")
                self._save_manifest()

                if not claims:
                    self._log_and_emit(log, "No claims kept. Ending run early.")
                    self.manifest["status"] = "early_exit"
                    self.manifest["topics"] = _extract_topics_lightweight(self.raw_text)
                    self._save_manifest()
                    self.emit("done", {"status": "early_exit", "message": "No claims kept"})
                    self.status = "done"
                    return
            else:
                # Mark review as skipped
                self.manifest.setdefault("timings", {})
                self.manifest["timings"]["review_claims"] = {"skipped": True}

            # Post-review group cleanup
            if groups:
                kept_ids = {c.claim_id for c in claims}
                cleaned_groups = []
                for g in groups:
                    g.claim_ids = [cid for cid in g.claim_ids if cid in kept_ids]
                    if len(g.claim_ids) >= 2:
                        cleaned_groups.append(g)
                if len(cleaned_groups) != len(groups):
                    self._log_and_emit(log, f"Groups after review cleanup: {len(cleaned_groups)} (was {len(groups)})")
                groups = cleaned_groups
                self.groups = groups
                self.manifest["counts"]["narrative_groups"] = len(groups)

            # ----- Stage 3: Retrieve evidence -----
            self._check_stop()
            self._log_and_emit(log, "Stage 3: retrieve evidence")
            s = self._stage_start("retrieve_evidence")

            budgets = cfg["budgets"]
            deny_domains = cfg.get("searx", {}).get("deny_domains") or []

            # Generate search queries via LLM (multi-query strategy)
            generated_queries = None
            if budgets.get("enable_query_generation", True):
                from app.tools.query_gen import generate_queries_batch
                model_query = cfg["ollama"].get("model_query_gen", cfg["ollama"]["model_extract"])
                temp_query = float(cfg["ollama"].get("temperature_query_gen", 0.3))
                num_q = int(budgets.get("queries_per_claim", 3))
                workers_q = int(budgets.get("query_gen_workers", 3))

                self._log_and_emit(log, f"Generating search queries ({num_q} per claim, {workers_q} workers)...")

                def _query_gen_progress(data):
                    self.emit("retrieve_progress", {
                        "claim_idx": data.get("current", 0),
                        "total_claims": data.get("total", len(claims)),
                        "claim_id": data.get("claim_id", ""),
                        "status": "generating_queries",
                    })

                generated_queries = generate_queries_batch(
                    cfg["ollama"]["base_url"], model_query, claims,
                    num_queries=num_q, temperature=temp_query, max_workers=workers_q,
                    progress_callback=_query_gen_progress,
                )
                self._log_and_emit(log, f"Generated search queries for {len(generated_queries)} claims")

                with open(os.path.join(self.outdir, "02d_queries.json"), "w", encoding="utf-8") as f:
                    json.dump(generated_queries, f, ensure_ascii=False, indent=2)
                self._add_artifact("generated_queries", "02d_queries.json")

            def _retrieve_progress(data):
                self.emit("retrieve_progress", data)
                status = data.get("status", "")
                claim_idx = data.get("claim_idx", 0)
                total = data.get("total_claims", 0)
                cid = data.get("claim_id", "")
                if status == "searching":
                    self._log_and_emit(log, f"Searching evidence for claim {claim_idx}/{total} ({cid})...")
                elif status == "done":
                    src = data.get("sources", 0)
                    snip = data.get("snippets", 0)
                    fail = data.get("failures", 0)
                    self._log_and_emit(log, f"Claim {claim_idx}/{total} done — {src} sources, {snip} snippets, {fail} failures")

            sources, snippets, evidence_by_claim, fetch_failures = retrieve_for_claims(
                cfg["searx"]["base_url"],
                claims,
                budgets,
                cfg["output"]["timezone"],
                snippets_per_source=budgets.get("snippets_per_source", 4),
                snippet_max_chars=budgets.get("snippet_max_chars", 1200),
                deny_domains=deny_domains,
                progress_callback=_retrieve_progress,
                generated_queries=generated_queries,
            )

            # Emit final counts BEFORE stage_end so UI counters
            # update while the detail panel is still visible
            self.manifest["counts"]["sources"] = len(sources)
            self.manifest["counts"]["snippets"] = len(snippets)
            self.manifest["counts"]["fetch_failures"] = len(fetch_failures)
            self.emit("progress", {
                "stage": "retrieve_evidence",
                "sources": len(sources),
                "snippets": len(snippets),
                "fetch_failures": len(fetch_failures),
            })

            self._stage_end("retrieve_evidence", s)

            with open(os.path.join(self.outdir, "03_sources.json"), "w", encoding="utf-8") as f:
                json.dump(sources, f, ensure_ascii=False, indent=2)
            with open(os.path.join(self.outdir, "03_snippets.json"), "w", encoding="utf-8") as f:
                json.dump(snippets, f, ensure_ascii=False, indent=2)
            with open(os.path.join(self.outdir, "04_evidence_by_claim.json"), "w", encoding="utf-8") as f:
                json.dump(evidence_by_claim, f, ensure_ascii=False, indent=2)
            with open(os.path.join(self.outdir, "04_fetch_failures.json"), "w", encoding="utf-8") as f:
                json.dump(fetch_failures, f, ensure_ascii=False, indent=2)

            self._add_artifact("sources", "03_sources.json")
            self._add_artifact("snippets", "03_snippets.json")
            self._add_artifact("evidence_by_claim", "04_evidence_by_claim.json")
            self._add_artifact("fetch_failures", "04_fetch_failures.json")

            hits = FETCH_STATS["cache_hit_ok"] + FETCH_STATS["cache_hit_fail"]
            misses = FETCH_STATS["cache_miss"]
            total_fetches = hits + misses
            hit_rate = (hits / total_fetches) if total_fetches else 0.0

            self.manifest["fetch_cache"] = {
                "cache_hit_ok": FETCH_STATS["cache_hit_ok"],
                "cache_hit_fail": FETCH_STATS["cache_hit_fail"],
                "cache_miss": FETCH_STATS["cache_miss"],
                "net_ok": FETCH_STATS["net_ok"],
                "net_fail": FETCH_STATS["net_fail"],
                "hit_rate": round(hit_rate, 4),
            }
            self._save_manifest()

            # ----- Stage 4: Check claims -----
            self._check_stop()
            self._log_and_emit(log, "Stage 4: check claims")
            s = self._stage_start("check_claims")

            from concurrent.futures import ThreadPoolExecutor, as_completed

            verify_workers = int(cfg["budgets"].get("verify_workers", 3))
            results_by_id = {}
            verified_count = 0

            def _verify_task(claim):
                t0v = time.time()
                v = verify_one(
                    cfg["ollama"]["base_url"],
                    cfg["ollama"]["model_verify"],
                    claim,
                    evidence_by_claim.get(claim.claim_id, {}),
                    self.outdir,
                    temperature=cfg["ollama"].get("temperature_verify", 0.0),
                    transcript_json=transcript_json,
                )
                return claim.claim_id, v, round(time.time() - t0v, 4)

            with ThreadPoolExecutor(max_workers=verify_workers) as executor:
                futures = {executor.submit(_verify_task, c): c for c in claims}
                for fut in as_completed(futures):
                    self._check_stop()
                    claim_id, v, sec = fut.result()
                    results_by_id[claim_id] = (v, sec)
                    verified_count += 1
                    self._log_and_emit(
                        log,
                        f"Verified {verified_count}/{len(claims)} "
                        f"claim_id={claim_id} rating={v.rating} conf={v.confidence} sec={sec}"
                    )
                    self.emit("verify_progress", {
                        "current": verified_count,
                        "total": len(claims),
                        "claim_id": claim_id,
                        "rating": str(v.rating),
                    })

            # Rebuild in original claim order
            verdicts = []
            per_claim_secs = []
            for c in claims:
                v, sec = results_by_id[c.claim_id]
                verdicts.append(v)
                per_claim_secs.append(sec)

            self.verdicts = verdicts
            self._stage_end("check_claims", s)

            with open(os.path.join(self.outdir, "05_verdicts.json"), "w", encoding="utf-8") as f:
                json.dump([v.model_dump() for v in verdicts], f, ensure_ascii=False, indent=2)

            self._add_artifact("verdicts", "05_verdicts.json")
            self.manifest["counts"]["verdicts"] = len(verdicts)

            if per_claim_secs:
                self.manifest["timings"].setdefault("check_claims", {})
                self.manifest["timings"]["check_claims"]["per_claim_sec"] = {
                    "min": round(min(per_claim_secs), 4),
                    "max": round(max(per_claim_secs), 4),
                    "avg": round(sum(per_claim_secs) / len(per_claim_secs), 4),
                    "n": len(per_claim_secs),
                }

            # Group verification (narrative-level)
            group_verdicts = []
            if groups:
                model_verify_group = cfg["ollama"].get("model_verify_group", cfg["ollama"]["model_verify"])
                self._log_and_emit(log, f"Verifying {len(groups)} narrative groups")
                for gi, g in enumerate(groups, 1):
                    self._check_stop()
                    self._log_and_emit(log, f"Group {gi}/{len(groups)}: {g.group_id} — {g.narrative_thesis[:60]}...")
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
                    self._log_and_emit(log, f"Group {g.group_id}: {gv.narrative_rating} (confidence={gv.narrative_confidence})")

                self.group_verdicts = group_verdicts
                with open(os.path.join(self.outdir, "05b_group_verdicts.json"), "w", encoding="utf-8") as f:
                    json.dump([gv.model_dump() for gv in group_verdicts], f, ensure_ascii=False, indent=2)
                self._add_artifact("group_verdicts", "05b_group_verdicts.json")

            self._save_manifest()

            # ----- Stage 5: Scorecard -----
            self._check_stop()
            self._log_and_emit(log, "Stage 5: scorecard")
            s = self._stage_start("scorecard")
            counts, red_flags, tiers = tally(verdicts)
            self._stage_end("scorecard", s)

            scorecard_md = f"""# Evident Scorecard

## Verdict counts
{json.dumps(counts, indent=2)}

## Source tiers used
{json.dumps(tiers, indent=2)}

## Red flags detected
{json.dumps(red_flags, indent=2)}
"""
            write_text(os.path.join(self.outdir, "06_scorecard.md"), scorecard_md)
            self._add_artifact("scorecard_md", "06_scorecard.md")

            self.manifest["scorecard"] = {
                "verdict_counts": counts,
                "tiers": tiers,
                "red_flags": red_flags,
            }
            self._save_manifest()

            # ----- Stage 6: Fact-check summary -----
            self._check_stop()
            self._log_and_emit(log, "Stage 6: fact-check summary")
            s = self._stage_start("fact_check_summary")
            writer_md = write_outline_and_script(
                cfg["ollama"]["base_url"],
                cfg["ollama"]["model_write"],
                transcript_json,
                verdicts,
                scorecard_md,
                claims,
                self.manifest["channel"]["raw"],
                groups=groups,
                group_verdicts=group_verdicts,
            )
            self._stage_end("fact_check_summary", s)

            write_text(os.path.join(self.outdir, "07_summary.md"), writer_md)
            self._add_artifact("writer_md", "07_summary.md")
            self.report_md = writer_md

            # ----- Finalize -----
            self.manifest["status"] = "ok"

            append_run_index(
                run_id=self.manifest["run_id"],
                input_file=self.manifest["infile"],
                outdir=self.manifest["outdir"],
                verdict_counts=self.manifest["scorecard"]["verdict_counts"],
                duration_sec=time.time() - t0,
            )

            topics = _extract_topics_lightweight(self.raw_text)
            self.manifest["topics"] = topics
            append_creator_profile_event(
                channel=self.manifest["channel"]["raw"],
                run_id=self.manifest["run_id"],
                verdict_counts=self.manifest["scorecard"]["verdict_counts"],
                red_flags=self.manifest["scorecard"].get("red_flags", []) if isinstance(self.manifest["scorecard"].get("red_flags"), list) else [],
                topics=topics,
                input_file=self.manifest["infile"],
                outdir=self.manifest["outdir"],
            )

            self._save_manifest()
            self._log_and_emit(log, f"RUN COMPLETE outdir={self.outdir}")

            # Emit done event BEFORE setting status so SSE generator
            # doesn't see "done" status while queue is still empty
            self.emit("done", {
                "status": "ok",
                "outdir": self.outdir,
            })
            self.status = "done"

        except PipelineCancelled:
            self.manifest["status"] = "cancelled"
            self._save_manifest()
            log.log("RUN CANCELLED by user")
            self._log_and_emit(log, "Run cancelled by user.")
            self.emit("done", {"status": "cancelled", "message": "Run cancelled by user"})
            self.status = "cancelled"

        except Exception as e:
            self.manifest["status"] = "error"
            self.manifest["error"] = {
                "type": type(e).__name__,
                "message": str(e),
                "traceback": traceback.format_exc(),
            }
            log.log(f"ERROR: {type(e).__name__}: {e}")
            log.log(self.manifest["error"]["traceback"])
            self._save_manifest()

            self.error_info = {
                "type": type(e).__name__,
                "message": str(e),
            }
            # Emit error event BEFORE setting status
            self.emit("error", {
                "message": f"{type(e).__name__}: {e}",
            })
            self.status = "error"

        finally:
            duration = time.time() - t0
            self.manifest["finished_utc"] = _utc_now_iso()
            self.manifest["duration_sec"] = round(duration, 3)
            self._save_manifest()
            log.log(f"RUN END status={self.manifest.get('status')} duration_sec={round(duration, 2)}")

    # ------------------------------------------------------------------
    # Review application
    # ------------------------------------------------------------------

    def _apply_review(self, claims, decisions):
        """
        Apply review decisions to claims list.
        Returns filtered/edited claims.
        """
        decisions_by_id = {d["claim_id"]: d for d in decisions}
        kept = []

        for c in claims:
            decision = decisions_by_id.get(c.claim_id)
            if decision is None:
                # No decision = keep as-is
                kept.append(c)
                continue

            action = decision.get("action", "keep")

            if action == "drop":
                continue

            if action == "edit":
                if decision.get("claim_text"):
                    c.claim_text = decision["claim_text"]
                if decision.get("claim_type"):
                    c.claim_type = decision["claim_type"]
                if decision.get("quote_from_transcript"):
                    c.quote_from_transcript = decision["quote_from_transcript"]

            kept.append(c)

        return kept
