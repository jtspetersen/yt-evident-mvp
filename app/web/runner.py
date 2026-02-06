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
from app.pipeline.verify_claims import verify_one
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
        "normalize_transcript",
        "extract_claims",
        "review_claims",
        "retrieve_evidence",
        "verify_claims",
        "scorecard",
        "write_outline_and_script",
    ]

    def __init__(self, cfg: dict, infile: str, raw_text: str,
                 channel: str = None, review_enabled: bool = False):
        self.cfg = cfg
        self.infile = infile
        self.raw_text = raw_text
        self.review_enabled = review_enabled

        inferred_channel = _infer_channel_from_filename(infile)
        self.channel = (channel or inferred_channel).strip() or "Unknown"

        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        transcript_name = os.path.basename(infile)
        transcript_base = re.sub(r"\.(txt|md)$", "", transcript_name, flags=re.IGNORECASE)
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
        self.verdicts = []
        self.manifest = {}
        self.report_md = None
        self.error_info = None

        # Review synchronization
        self.review_result = None
        self._review_event = threading.Event()

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
                "review_mode": self.review_enabled,
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
            "stage_order": self.STAGE_ORDER,
            "error": None,
            "artifacts": {},
            "outputs": [],
        }

        self._add_artifact("manifest", "run.json")
        self._add_artifact("log", "run.log")

        # Save raw transcript
        raw_copy_name = "00_transcript.raw.txt"
        with open(os.path.join(self.outdir, raw_copy_name), "w", encoding="utf-8") as f:
            f.write(self.raw_text)
        self._add_artifact("transcript_raw", raw_copy_name)
        self._save_manifest()

        try:
            self._log_and_emit(log, f"RUN START run_id={self.run_id} channel={self.channel}")

            # ----- Stage 1: Normalize transcript -----
            self._log_and_emit(log, "Stage 1: normalize transcript")
            s = self._stage_start("normalize_transcript")
            transcript_json = normalize_transcript(self.raw_text)
            write_json(os.path.join(self.outdir, "01_transcript.normalized.json"), transcript_json)
            self._stage_end("normalize_transcript", s)
            self._add_artifact("transcript_normalized", "01_transcript.normalized.json")
            self._save_manifest()

            # ----- Stage 2: Extract claims -----
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
                temperature=cfg["ollama"].get("temperature_extract", 0.1),
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

            # ----- Stage 2b: Review (pause if enabled) -----
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

            # ----- Stage 3: Retrieve evidence -----
            self._log_and_emit(log, "Stage 3: retrieve evidence")
            s = self._stage_start("retrieve_evidence")

            budgets = cfg["budgets"]
            deny_domains = cfg.get("searx", {}).get("deny_domains") or []

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

            # ----- Stage 4: Verify claims -----
            self._log_and_emit(log, "Stage 4: verify claims")
            s = self._stage_start("verify_claims")

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
            self._stage_end("verify_claims", s)

            with open(os.path.join(self.outdir, "05_verdicts.json"), "w", encoding="utf-8") as f:
                json.dump([v.model_dump() for v in verdicts], f, ensure_ascii=False, indent=2)

            self._add_artifact("verdicts", "05_verdicts.json")
            self.manifest["counts"]["verdicts"] = len(verdicts)

            if per_claim_secs:
                self.manifest["timings"].setdefault("verify_claims", {})
                self.manifest["timings"]["verify_claims"]["per_claim_sec"] = {
                    "min": round(min(per_claim_secs), 4),
                    "max": round(max(per_claim_secs), 4),
                    "avg": round(sum(per_claim_secs) / len(per_claim_secs), 4),
                    "n": len(per_claim_secs),
                }

            self._save_manifest()

            # ----- Stage 5: Scorecard -----
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

            # ----- Stage 6: Write outline + script -----
            self._log_and_emit(log, "Stage 6: write outline + script")
            s = self._stage_start("write_outline_and_script")
            writer_md = write_outline_and_script(
                cfg["ollama"]["base_url"],
                cfg["ollama"]["model_write"],
                transcript_json,
                verdicts,
                scorecard_md,
                claims,
                self.manifest["channel"]["raw"],
            )
            self._stage_end("write_outline_and_script", s)

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
