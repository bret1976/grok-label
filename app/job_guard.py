"""Job guard (job-guard-v1): protect the paid Grok vision pipeline on public uploads.

Backend only. Every upload to ``POST /api/jobs`` can fan out into
``frames x (1 + grid^2)`` Grok vision calls (about 405 calls for a 20s clip),
so before a job is queued this module:

- caps upload size while streaming (``MAX_UPLOAD_MB``, default 300) -> 413;
- probes the file with ffprobe and rejects anything without a real video stream
  or a usable duration -> 400 (validate by probing, not by file extension);
- caps clip length (``MAX_CLIP_SECONDS``, default 180) -> 400;
- reuses a finished job when the exact same bytes were already labeled with the
  same sample rate, tile grid, and model (SHA-256 content address) -> no new spend;
- limits concurrently queued/running jobs (``MAX_ACTIVE_JOBS``, default 2) -> 429;
- marks a crashed job as ``error`` instead of leaving it "running" forever.

Soft kill switch: ``JOB_GUARD=0`` turns off the size/probe/length/reuse/active
checks (crash capture stays on).

Ideas, not code, from: SHA-256 upload dedupe before enqueue (common FastAPI job
pattern, e.g. InfantLab/VideoAnnotator chunked upload + job ids), "validate video
uploads with ffprobe, not extensions" (Rendobar), Cloudflare Stream
``maxDurationSeconds``, and pre-call LLM budget guards (r/ArtificialInteligence,
Amitcoh1/agentbreaker). Original code; nothing vendored.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

JOB_GUARD_ID = "job-guard-v1"

_LOCK = threading.Lock()
COUNTERS: dict[str, int] = {
    "accepted": 0,
    "reused": 0,
    "blocked_too_large": 0,
    "blocked_not_video": 0,
    "blocked_too_long": 0,
    "blocked_busy": 0,
    "crashed_marked_error": 0,
    "grok_calls_saved_estimate": 0,
}


class GuardReject(Exception):
    def __init__(self, status: int, reason: str, detail: str):
        super().__init__(detail)
        self.status = status
        self.reason = reason
        self.detail = detail


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name) or default)
    except (TypeError, ValueError):
        return default


def enabled() -> bool:
    return str(os.environ.get("JOB_GUARD", "1")).strip().lower() not in {"0", "false", "off", "no"}


def limits() -> dict[str, float]:
    return {
        "max_upload_mb": _env_float("MAX_UPLOAD_MB", 300),
        "max_clip_seconds": _env_float("MAX_CLIP_SECONDS", 180),
        "max_active_jobs": int(_env_float("MAX_ACTIVE_JOBS", 2)),
        "stale_minutes": _env_float("JOB_STALE_MINUTES", 30),
    }


def bump(key: str, n: int = 1) -> None:
    with _LOCK:
        COUNTERS[key] = COUNTERS.get(key, 0) + n


def max_upload_bytes() -> int:
    return int(limits()["max_upload_mb"] * 1024 * 1024)


def estimate_calls(duration: float, sample_fps: float, grid: int) -> int:
    frames = max(1, math.ceil(max(duration, 0.0) * max(sample_fps, 0.0)))
    per_frame = 1 + (grid * grid if grid > 1 else 0)
    return frames * per_frame


def probe_video(path: Path) -> dict[str, Any]:
    """ffprobe the file. Returns {ok, reason, duration, width, height, codec}."""
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-print_format",
                "json",
                "-show_format",
                "-show_streams",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except FileNotFoundError:
        # No ffprobe in this environment: fail open (the pipeline itself needs ffmpeg).
        return {"ok": True, "reason": "probe_unavailable", "duration": None}
    except subprocess.TimeoutExpired:
        return {"ok": False, "reason": "probe_timeout", "duration": None}
    if result.returncode != 0:
        return {"ok": False, "reason": "unreadable", "duration": None}
    try:
        data = json.loads(result.stdout or "{}")
    except json.JSONDecodeError:
        return {"ok": False, "reason": "unreadable", "duration": None}
    streams = data.get("streams") or []
    video = next(
        (
            s
            for s in streams
            if s.get("codec_type") == "video" and not (s.get("disposition") or {}).get("attached_pic")
        ),
        None,
    )
    if not video:
        return {"ok": False, "reason": "no_video_stream", "duration": None}
    fmt = data.get("format") or {}
    try:
        duration = float(fmt.get("duration") or video.get("duration") or 0)
    except (TypeError, ValueError):
        duration = 0.0
    if fmt.get("format_name", "").startswith("image2") or not duration > 0:
        return {"ok": False, "reason": "no_duration", "duration": duration or None}
    return {
        "ok": True,
        "reason": "ok",
        "duration": round(duration, 3),
        "width": video.get("width"),
        "height": video.get("height"),
        "codec": video.get("codec_name"),
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(content_sha: str, *, sample_fps: float, grid: int, model: str) -> str:
    raw = f"{content_sha}|fps={float(sample_fps):g}|grid={int(grid)}|model={model}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _iter_jobs(jobs_root: Path):
    if not jobs_root.exists():
        return
    for path in jobs_root.glob("*/job.json"):
        try:
            yield path, json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue


def find_reusable(jobs_root: Path, fp: str) -> dict[str, Any] | None:
    best = None
    best_mtime = -1.0
    for path, job in _iter_jobs(jobs_root):
        if job.get("fingerprint") != fp or job.get("status") != "done" or not job.get("frames"):
            continue
        if not Path(str(job.get("video_path") or "")).exists():
            continue
        mtime = path.stat().st_mtime
        if mtime > best_mtime:
            best, best_mtime = job, mtime
    return best


def active_jobs(jobs_root: Path) -> list[str]:
    stale = limits()["stale_minutes"] * 60
    now = time.time()
    out = []
    for path, job in _iter_jobs(jobs_root):
        if job.get("status") not in {"queued", "running"}:
            continue
        try:
            if now - path.stat().st_mtime > stale:
                continue  # orphaned by a restart; don't let it block new work
        except OSError:
            continue
        out.append(str(job.get("id")))
    return out


def check_upload(
    path: Path,
    *,
    jobs_root: Path,
    sample_fps: float,
    grid: int,
    model: str,
) -> dict[str, Any]:
    """Run probe/length/reuse/busy checks. Raises GuardReject. Returns guard info.

    Result keys: fingerprint, content_sha256, probe, estimated_grok_calls, reuse_job (or None).
    """
    lim = limits()
    probe = probe_video(path)
    if not probe.get("ok"):
        bump("blocked_not_video")
        raise GuardReject(
            400,
            probe.get("reason") or "not_video",
            "That file doesn't look like a playable video clip. Upload an MP4, MOV, or WebM.",
        )
    duration = probe.get("duration")
    if duration is not None and duration > lim["max_clip_seconds"]:
        bump("blocked_too_long")
        raise GuardReject(
            400,
            "clip_too_long",
            f"Clip is {duration:.0f}s; the limit is {lim['max_clip_seconds']:.0f}s. Trim it and try again.",
        )
    content_sha = file_sha256(path)
    fp = fingerprint(content_sha, sample_fps=sample_fps, grid=grid, model=model)
    est = estimate_calls(duration or 0.0, sample_fps, grid) if duration else None
    info: dict[str, Any] = {
        "id": JOB_GUARD_ID,
        "fingerprint": fp,
        "content_sha256": content_sha,
        "probe": probe,
        "estimated_grok_calls": est,
        "reuse_job": None,
    }
    reuse = find_reusable(jobs_root, fp)
    if reuse:
        bump("reused")
        if est:
            bump("grok_calls_saved_estimate", est)
        info["reuse_job"] = reuse
        return info
    busy = active_jobs(jobs_root)
    if len(busy) >= lim["max_active_jobs"]:
        bump("blocked_busy")
        raise GuardReject(
            429,
            "busy",
            f"{len(busy)} clips are already labeling. Try again in a minute.",
        )
    return info


def summary(jobs_root: Path) -> dict[str, Any]:
    with _LOCK:
        counters = dict(COUNTERS)
    return {
        "id": JOB_GUARD_ID,
        "enabled": enabled(),
        "limits": limits(),
        "active_jobs": len(active_jobs(jobs_root)),
        "counters": counters,
        "notes": "Counters are per process and reset on deploy. JOB_GUARD=0 disables checks; crash capture stays on.",
    }
