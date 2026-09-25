from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

from app.frames import extract_frames
from app.grok_vision import label_frame
from app.tracker import assign_tracks, nms, stats

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
JOBS = DATA / "jobs"


def job_dir(job_id: str) -> Path:
    path = JOBS / job_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_job(job_id: str) -> dict[str, Any] | None:
    path = job_dir(job_id) / "job.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def save_job(job: dict[str, Any]) -> dict[str, Any]:
    path = job_dir(job["id"]) / "job.json"
    path.write_text(json.dumps(job, indent=2), encoding="utf-8")
    return job


def log(job: dict[str, Any], text: str, kind: str = "note") -> None:
    job.setdefault("log", []).append({"t": time.time(), "kind": kind, "text": text})
    save_job(job)


async def run_job(job_id: str) -> None:
    job = load_job(job_id)
    if not job:
        return
    job["status"] = "running"
    job["stage"] = "generate"
    log(job, "Grok accepted the request. Extracting CCTV frames.", "tool")
    video = Path(job["video_path"])
    frames_dir = job_dir(job_id) / "frames"
    sample_fps = float(job.get("sample_fps") or os.environ.get("SAMPLE_FPS") or 4)
    grid = int(job.get("grid") or os.environ.get("TILE_GRID") or 2)
    model = job.get("model") or os.environ.get("GROK_MODEL") or "grok-4.20-0309-non-reasoning"
    concurrency = int(os.environ.get("MAX_CONCURRENCY") or 4)

    frames = extract_frames(video, frames_dir, sample_fps)
    job["duration"] = frames and json.loads((frames_dir / "meta.json").read_text())["duration"]
    job["frame_count"] = len(frames)
    job["stage"] = "annotate"
    log(
        job,
        f"Sampled {len(frames)} frames at {sample_fps} fps. "
        f"Labeling each frame with Grok ({model}), including a {grid}×{grid} tile pass for small parcels.",
        "tool",
    )
    save_job(job)

    sem = asyncio.Semaphore(concurrency)

    async def one(frame: dict[str, Any]) -> dict[str, Any]:
        async with sem:
            dets = await label_frame(Path(frame["path"]), model=model, grid=grid)
            frame["detections"] = nms(dets)
            return frame

    done = 0
    for batch_start in range(0, len(frames), concurrency):
        chunk = frames[batch_start : batch_start + concurrency]
        await asyncio.gather(*(one(frame) for frame in chunk))
        done += len(chunk)
        peak = max((len(f.get("detections") or []) for f in frames[:done]), default=0)
        job["progress"] = round(done / len(frames), 3)
        log(job, f"Labeled {done}/{len(frames)} frames. Peak detections in a frame: {peak}.", "note")
        save_job(job)

    job["stage"] = "inspect"
    tracks = assign_tracks(frames)
    job["frames"] = [
        {
            "index": f["index"],
            "t": f["t"],
            "name": f["name"],
            "detections": f.get("detections") or [],
        }
        for f in frames
    ]
    job["tracks"] = tracks["tracks"]
    job["class_counts"] = tracks["class_counts"]
    job["stats"] = stats(job["frames"], job["tracks"])
    job["tracker"] = tracks.get("tracker") or {}
    tk = job["tracker"]
    log(
        job,
        f"BYTE tracker: {tk.get('refound_after_miss', 0)} tracks re-found after a missed frame, "
        f"{tk.get('rescued_low_score', 0)} low-confidence boxes kept on their track, "
        f"{tk.get('gap_filled_boxes', 0)} occlusion gaps filled.",
        "tool",
    )
    job["status"] = "done"
    job["progress"] = 1
    log(
        job,
        f"Completed pass: {job['stats']['annotations']} detections across {len(frames)} frames, "
        f"{job['stats']['all']} unique tracks. Inspect overlays, trails, and interpolation.",
        "done",
    )
    save_job(job)


def new_job(video_path: Path, *, sample_fps: float, grid: int, model: str) -> dict[str, Any]:
    job_id = uuid.uuid4().hex[:10]
    dest = job_dir(job_id) / "source.mp4"
    dest.write_bytes(video_path.read_bytes())
    job = {
        "id": job_id,
        "status": "queued",
        "stage": "generate",
        "progress": 0,
        "video_path": str(dest),
        "sample_fps": sample_fps,
        "grid": grid,
        "model": model,
        "log": [],
        "frames": [],
        "tracks": [],
        "stats": {},
    }
    log(job, "Queued your clip. Grok vision will detect persons, boxes, totes, and carts, then stitch tracks.", "note")
    return save_job(job)
