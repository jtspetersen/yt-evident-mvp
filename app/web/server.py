# app/web/server.py
"""
FastAPI web server for Evident Video Fact Checker.

Provides a browser UI for uploading transcripts, monitoring pipeline
progress via SSE, reviewing claims, and viewing reports.

Run with: python -m app.web.server
"""
import asyncio
import json
import os
import queue
from pathlib import Path

import markdown
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, RedirectResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.main import load_config
from app.web.runner import PipelineRunner, get_runner, list_runners

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

WEB_DIR = Path(__file__).parent
TEMPLATES_DIR = WEB_DIR / "templates"
STATIC_DIR = WEB_DIR / "static"

app = FastAPI(title="Evident Video Fact Checker")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Upload form — main landing page."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload")
async def upload(
    request: Request,
    transcript: UploadFile = File(...),
    channel: str = Form(""),
    review: bool = Form(False),
):
    """Accept transcript upload, start pipeline, redirect to progress."""
    # Ensure inbox dir exists
    os.makedirs("inbox", exist_ok=True)

    # Save uploaded file
    filename = transcript.filename or "upload.txt"
    infile_path = os.path.join("inbox", filename)
    content = await transcript.read()
    raw_text = content.decode("utf-8", errors="replace")
    with open(infile_path, "w", encoding="utf-8") as f:
        f.write(raw_text)

    # Load config and start pipeline
    cfg = load_config()
    runner = PipelineRunner(
        cfg=cfg,
        infile=infile_path,
        raw_text=raw_text,
        channel=channel.strip() or None,
        review_enabled=review,
    )
    runner.start()

    return RedirectResponse(url=f"/run/{runner.run_id}", status_code=303)


@app.get("/run/{run_id}", response_class=HTMLResponse)
async def run_progress(request: Request, run_id: str):
    """Progress dashboard — auto-updates via SSE."""
    runner = get_runner(run_id)
    if not runner:
        return templates.TemplateResponse("error.html", {
            "request": request,
            "message": f"Run {run_id} not found.",
        }, status_code=404)

    # If already done, redirect to report
    if runner.status == "done" and runner.manifest.get("status") == "ok":
        return RedirectResponse(url=f"/run/{run_id}/report", status_code=303)

    return templates.TemplateResponse("progress.html", {
        "request": request,
        "run_id": run_id,
        "runner": runner,
    })


@app.get("/run/{run_id}/events")
async def run_events(run_id: str):
    """SSE stream — stage updates, log lines, progress."""
    runner = get_runner(run_id)
    if not runner:
        return StreamingResponse(
            _sse_error("Run not found"),
            media_type="text/event-stream",
        )

    async def event_generator():
        while True:
            # Drain all queued events in a batch
            batch = []
            while True:
                try:
                    batch.append(runner.events.get_nowait())
                except queue.Empty:
                    break

            # Yield all events from this batch
            for event in batch:
                yield f"event: {event['event']}\ndata: {event['data']}\n\n"

            # If pipeline finished AND queue is empty, do one final drain then stop
            if runner.status in ("done", "error", "cancelled") and not batch:
                # Brief pause for any last events the thread may emit
                await asyncio.sleep(0.15)
                while True:
                    try:
                        event = runner.events.get_nowait()
                        yield f"event: {event['event']}\ndata: {event['data']}\n\n"
                    except queue.Empty:
                        break
                yield f"event: stream_end\ndata: {json.dumps({'status': runner.status})}\n\n"
                break

            # Send SSE comment as keepalive to prevent connection drop
            yield ": heartbeat\n\n"
            await asyncio.sleep(0.4)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/run/{run_id}/review", response_class=HTMLResponse)
async def review_page(request: Request, run_id: str):
    """Claim review interface — shown when pipeline pauses for review."""
    runner = get_runner(run_id)
    if not runner:
        return templates.TemplateResponse("error.html", {
            "request": request,
            "message": f"Run {run_id} not found.",
        }, status_code=404)

    if runner.status != "review":
        return RedirectResponse(url=f"/run/{run_id}", status_code=303)

    claims_data = [c.model_dump() for c in runner.claims]

    return templates.TemplateResponse("review.html", {
        "request": request,
        "run_id": run_id,
        "claims": claims_data,
    })


@app.post("/run/{run_id}/review")
async def submit_review(request: Request, run_id: str):
    """Submit review decisions — JSON array of {claim_id, action, ...}."""
    runner = get_runner(run_id)
    if not runner:
        return {"error": "Run not found"}, 404

    body = await request.json()
    decisions = body if isinstance(body, list) else body.get("decisions", [])
    runner.submit_review(decisions)

    return RedirectResponse(url=f"/run/{run_id}", status_code=303)


@app.post("/run/{run_id}/stop")
async def stop_run(run_id: str):
    """Stop a running pipeline."""
    runner = get_runner(run_id)
    if not runner:
        return RedirectResponse(url=f"/run/{run_id}", status_code=303)
    runner.stop()
    return RedirectResponse(url=f"/run/{run_id}", status_code=303)


@app.get("/run/{run_id}/report", response_class=HTMLResponse)
async def report_page(request: Request, run_id: str):
    """Final report display — rendered markdown."""
    runner = get_runner(run_id)

    report_html = ""
    manifest = {}
    scorecard = {}

    if runner:
        manifest = runner.manifest
        scorecard = manifest.get("scorecard", {})
        if runner.report_md:
            report_html = markdown.markdown(
                runner.report_md,
                extensions=["tables", "fenced_code"],
            )
    else:
        # Try loading from disk (for past runs)
        report_html, manifest, scorecard = _load_report_from_disk(run_id)

    if not report_html:
        return templates.TemplateResponse("error.html", {
            "request": request,
            "message": f"Report for run {run_id} not available yet.",
        }, status_code=404)

    return templates.TemplateResponse("report.html", {
        "request": request,
        "run_id": run_id,
        "report_html": report_html,
        "manifest": manifest,
        "scorecard": scorecard,
    })


@app.get("/run/{run_id}/artifact/{name}")
async def download_artifact(run_id: str, name: str):
    """Download a raw artifact file from a run."""
    runner = get_runner(run_id)
    if runner:
        outdir = runner.outdir
    else:
        outdir = _find_run_dir(run_id)

    if not outdir:
        return {"error": "Run not found"}, 404

    file_path = os.path.join(outdir, name)
    if not os.path.isfile(file_path):
        return {"error": "Artifact not found"}, 404

    return FileResponse(file_path, filename=name)


@app.get("/runs", response_class=HTMLResponse)
async def runs_list(request: Request):
    """Past runs list — reads from store/run_index.jsonl."""
    runs = []
    index_path = os.path.join("store", "run_index.jsonl")
    if os.path.isfile(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        runs.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    # Most recent first
    runs.reverse()

    return templates.TemplateResponse("runs.html", {
        "request": request,
        "runs": runs,
    })


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _sse_error(message: str):
    yield f"event: error\ndata: {json.dumps({'message': message})}\n\n"


def _find_run_dir(run_id: str) -> str | None:
    """Find a run directory by run_id prefix in runs/."""
    runs_dir = "runs"
    if not os.path.isdir(runs_dir):
        return None
    for entry in os.listdir(runs_dir):
        if entry.startswith(run_id):
            path = os.path.join(runs_dir, entry)
            if os.path.isdir(path):
                return path
    return None


def _load_report_from_disk(run_id: str) -> tuple:
    """Load report from disk for past runs not in memory."""
    outdir = _find_run_dir(run_id)
    if not outdir:
        return "", {}, {}

    # Load manifest
    manifest = {}
    manifest_path = os.path.join(outdir, "run.json")
    if os.path.isfile(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

    scorecard = manifest.get("scorecard", {})

    # Load report markdown
    report_md_path = os.path.join(outdir, "07_summary.md")
    if os.path.isfile(report_md_path):
        with open(report_md_path, "r", encoding="utf-8") as f:
            report_md = f.read()
        report_html = markdown.markdown(
            report_md,
            extensions=["tables", "fenced_code"],
        )
        return report_html, manifest, scorecard

    return "", manifest, scorecard


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    import uvicorn
    print("Starting Evident Web UI at http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")


if __name__ == "__main__":
    main()
