from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.grok_vision import resolve_api_key
from app.pipeline import JOBS, load_job, new_job, run_job
from app.tracker import interpolate_box

load_dotenv()


def _has_key() -> bool:
    try:
        return bool(resolve_api_key())
    except RuntimeError:
        return False

ROOT = Path(__file__).resolve().parent.parent
STATIC = ROOT / "static"
RESEARCH = ROOT / "research"


@asynccontextmanager
async def lifespan(app: FastAPI):
    JOBS.mkdir(parents=True, exist_ok=True)
    yield


app = FastAPI(title="Grok Label", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")


@app.get("/")
async def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/health")
async def health():
    return {
        "ok": True,
        "product": "Grok Label",
        "model": os.environ.get("GROK_MODEL", "grok-4.20-0309-non-reasoning"),
        "has_key": True if _has_key() else False,
        "upload": True,
    }


def _start(background: BackgroundTasks, video: Path):
    job = new_job(
        video,
        sample_fps=float(os.environ.get("SAMPLE_FPS") or 4),
        grid=int(os.environ.get("TILE_GRID") or 2),
        model=os.environ.get("GROK_MODEL") or "grok-4.20-0309-non-reasoning",
    )
    background.add_task(run_job, job["id"])
    return {"ok": True, "job_id": job["id"], "job": public_job(job)}


@app.post("/api/jobs")
async def create_job(background: BackgroundTasks, file: UploadFile = File(...)):
    suffix = Path(file.filename or "clip.mp4").suffix or ".mp4"
    tmp = JOBS / f"upload-{Path(file.filename or 'clip').stem}{suffix}"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_bytes(await file.read())
    return _start(background, tmp)





def public_job(job: dict) -> dict:
    return {
        "id": job.get("id"),
        "status": job.get("status"),
        "stage": job.get("stage"),
        "progress": job.get("progress") or 0,
        "duration": job.get("duration"),
        "frame_count": job.get("frame_count") or len(job.get("frames") or []),
        "stats": job.get("stats") or {},
        "class_counts": job.get("class_counts") or {},
        "log": job.get("log") or [],
        "model": job.get("model"),
        "has_frames": bool(job.get("frames")),
    }


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    job = load_job(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    return public_job(job)


@app.get("/api/jobs/{job_id}/annotations")
async def annotations(job_id: str):
    job = load_job(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    return {
        "frames": job.get("frames") or [],
        "tracks": job.get("tracks") or [],
        "stats": job.get("stats") or {},
        "duration": job.get("duration") or 0,
    }


@app.get("/api/jobs/{job_id}/overlay")
async def overlay(job_id: str, t: float = 0.0):
    job = load_job(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    frames = job.get("frames") or []
    if not frames:
        return {"t": t, "detections": [], "source": "none"}
    times = [f["t"] for f in frames]
    if t <= times[0]:
        return {"t": t, "detections": frames[0].get("detections") or [], "source": "detected"}
    if t >= times[-1]:
        return {"t": t, "detections": frames[-1].get("detections") or [], "source": "detected"}
    lo = 0
    for i, ft in enumerate(times):
        if ft <= t:
            lo = i
    hi = min(lo + 1, len(frames) - 1)
    a, b = frames[lo], frames[hi]
    span = (b["t"] - a["t"]) or 1e-6
    alpha = (t - a["t"]) / span
    by_id_a = {d["track_id"]: d for d in a.get("detections") or [] if d.get("track_id")}
    by_id_b = {d["track_id"]: d for d in b.get("detections") or [] if d.get("track_id")}
    mixed = []
    for tid, da in by_id_a.items():
        db = by_id_b.get(tid)
        mixed.append(interpolate_box(da, db, alpha) if db else {**da, "source": "detected"})
    for tid, db in by_id_b.items():
        if tid not in by_id_a:
            mixed.append({**db, "source": "detected"})
    return {"t": t, "detections": mixed, "source": "interpolated"}


@app.get("/api/jobs/{job_id}/video")
async def job_video(job_id: str):
    job = load_job(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    path = Path(job["video_path"])
    if not path.exists():
        raise HTTPException(404, "video missing")
    return FileResponse(path, media_type="video/mp4")


@app.get("/api/jobs/{job_id}/frame/{name}")
async def job_frame(job_id: str, name: str):
    path = JOBS / job_id / "frames" / Path(name).name
    if not path.exists():
        raise HTTPException(404, "frame missing")
    return FileResponse(path, media_type="image/jpeg")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="127.0.0.1", port=int(os.environ.get("PORT", "8787")), reload=True)
