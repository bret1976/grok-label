"""Stale job watch (stale-job-watch-v1): reaper for stuck label jobs.

Backend only. Grok Label runs labeling in FastAPI BackgroundTasks. A Railway
deploy, process crash, or OOM leaves jobs as ``queued`` / ``running`` forever —
job-guard-v1 already *skips* those when counting active slots, but never closes
them, so the ops UI and job JSON stay lying about work that will never finish.

This pack:

- sweeps job.json files stuck in ``queued`` / ``running`` past
  ``STALE_JOB_WATCH_MINUTES`` (default 30, same idea as JOB_STALE_MINUTES);
- marks them ``error`` with reason ``stale_timeout`` (dead-letter style terminal
  state — no auto-retry of paid Grok calls; operator can re-upload);
- exposes counts-only ``GET /api/stale-job-watch/summary``;
- runs on startup, before new uploads, and when the summary is fetched.

Soft kill switch: ``STALE_JOB_WATCH=0`` restores the old path (no reaping).

Ideas, not code, from: imgeaslikok/deterministic-job-pipeline (MIT, stuck
RUNNING reaper → reset / DLQ), changhyeon363/pqrun (Apache-2.0, stale reaper
loop), balyakin/pgrelay (dead_letter after exhausted attempts), and Reddit
r/FastAPI threads on BackgroundTasks dying on deploy without a terminal state.
Original pure-Python; nothing vendored; no new dependencies.
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

STALE_JOB_WATCH_ID = "stale-job-watch-v1"

_LOCK = threading.Lock()
COUNTERS: dict[str, int] = {
    "sweeps": 0,
    "marked_stale": 0,
    "already_terminal": 0,
    "skipped_fresh": 0,
    "read_errors": 0,
}


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name) or default)
    except (TypeError, ValueError):
        return default


def enabled() -> bool:
    return str(os.environ.get("STALE_JOB_WATCH", "1")).strip().lower() not in {
        "0",
        "false",
        "off",
        "no",
    }


def stale_seconds() -> float:
    # Prefer pack-specific env; fall back to job-guard's JOB_STALE_MINUTES.
    if os.environ.get("STALE_JOB_WATCH_MINUTES") not in (None, ""):
        return _env_float("STALE_JOB_WATCH_MINUTES", 30) * 60
    return _env_float("JOB_STALE_MINUTES", 30) * 60


def bump(key: str, n: int = 1) -> None:
    with _LOCK:
        COUNTERS[key] = COUNTERS.get(key, 0) + n


def _job_age_seconds(path: Path, job: dict[str, Any], now: float) -> float:
    """Age from last progress signal: job.json mtime, else started/updated fields."""
    try:
        mtime = path.stat().st_mtime
    except OSError:
        mtime = now
    candidates = [mtime]
    for key in ("updated_at", "started_at", "created_at"):
        raw = job.get(key)
        if not isinstance(raw, (int, float)) or raw <= 0:
            continue
        ts = float(raw)
        if ts > 1e12:  # epoch milliseconds
            ts = ts / 1000.0
        candidates.append(ts)
    # Prefer the *newest* signal so a progressing job (mtime refreshed by save_job) stays fresh.
    newest = max(candidates)
    return max(0.0, now - newest)


def _mark_stale(path: Path, job: dict[str, Any], age_s: float, limit_s: float) -> bool:
    """Persist terminal error. Returns True if we wrote a change."""
    if job.get("status") not in {"queued", "running"}:
        return False
    minutes = max(1, int(round(age_s / 60)))
    limit_m = max(1, int(round(limit_s / 60)))
    job["status"] = "error"
    job["error"] = "stale_timeout"
    job["stale"] = {
        "id": STALE_JOB_WATCH_ID,
        "age_seconds": round(age_s, 1),
        "limit_seconds": limit_s,
        "marked_at": time.time(),
    }
    job.setdefault("log", []).append(
        {
            "t": time.time(),
            "kind": "error",
            "text": (
                f"Labeling stopped: job was still {job.get('stage') or 'in progress'} "
                f"after {minutes} min (limit {limit_m} min). Marked stale — re-upload to retry."
            ),
        }
    )
    path.write_text(json.dumps(job, indent=2), encoding="utf-8")
    return True


def sweep(jobs_root: Path, *, now: float | None = None) -> dict[str, Any]:
    """Reap stuck queued/running jobs. Safe no-op when disabled or root missing."""
    result: dict[str, Any] = {
        "id": STALE_JOB_WATCH_ID,
        "enabled": enabled(),
        "marked": [],
        "scanned": 0,
    }
    if not enabled():
        return result
    bump("sweeps")
    root = Path(jobs_root)
    if not root.exists():
        return result
    ts = time.time() if now is None else now
    limit = stale_seconds()
    marked: list[dict[str, Any]] = []
    for path in root.glob("*/job.json"):
        result["scanned"] += 1
        try:
            job = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            bump("read_errors")
            continue
        status = job.get("status")
        if status not in {"queued", "running"}:
            bump("already_terminal")
            continue
        age = _job_age_seconds(path, job, ts)
        if age < limit:
            bump("skipped_fresh")
            continue
        if _mark_stale(path, job, age, limit):
            bump("marked_stale")
            marked.append(
                {
                    "id": job.get("id"),
                    "was": status,
                    "age_seconds": round(age, 1),
                }
            )
    result["marked"] = marked
    return result


def summary(jobs_root: Path) -> dict[str, Any]:
    """Counts-only summary; runs a sweep first so the dashboard self-heals."""
    sweep_result = sweep(jobs_root)
    open_stuck = 0
    open_fresh = 0
    errors_stale = 0
    root = Path(jobs_root)
    now = time.time()
    limit = stale_seconds()
    if root.exists():
        for path in root.glob("*/job.json"):
            try:
                job = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            status = job.get("status")
            if status in {"queued", "running"}:
                age = _job_age_seconds(path, job, now)
                if age >= limit:
                    open_stuck += 1
                else:
                    open_fresh += 1
            elif status == "error" and (
                job.get("error") == "stale_timeout" or (job.get("stale") or {}).get("id") == STALE_JOB_WATCH_ID
            ):
                errors_stale += 1
    with _LOCK:
        counters = dict(COUNTERS)
    return {
        "id": STALE_JOB_WATCH_ID,
        "enabled": enabled(),
        "limits": {
            "stale_minutes": stale_seconds() / 60,
            "stale_seconds": stale_seconds(),
        },
        "open_queued_or_running": open_fresh + open_stuck,
        "open_fresh": open_fresh,
        "open_stuck_should_be_zero": open_stuck,
        "marked_stale_total_on_disk": errors_stale,
        "last_sweep_marked": len(sweep_result.get("marked") or []),
        "counters": counters,
        "notes": (
            "Marks queued/running jobs older than the limit as error/stale_timeout. "
            "STALE_JOB_WATCH=0 disables. Does not auto-retry paid Grok calls."
        ),
    }
