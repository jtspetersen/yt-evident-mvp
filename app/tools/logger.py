# app/tools/logger.py
import os
from datetime import datetime
from typing import Optional


class RunLogger:
    """
    Simple run logger:
    - prints to stdout
    - optionally appends to a logfile (e.g., runs/<run>/run.log)
    """

    def __init__(self, run_id: str, logfile_path: Optional[str] = None):
        self.run_id = run_id
        self.logfile_path = logfile_path

        if self.logfile_path:
            os.makedirs(os.path.dirname(self.logfile_path), exist_ok=True)
            # Touch early so it exists even if we crash later
            with open(self.logfile_path, "a", encoding="utf-8") as f:
                f.write("")

    def log(self, msg: str) -> None:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line)
        if self.logfile_path:
            with open(self.logfile_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")


def make_run_logger(run_id: str, outdir: str | None = None) -> RunLogger:
    logfile_path = os.path.join(outdir, "run.log") if outdir else None
    return RunLogger(run_id=run_id, logfile_path=logfile_path)