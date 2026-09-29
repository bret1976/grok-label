from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from typing import Any

from fastapi import BackgroundTasks, Body, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from app.export import (
    CATEGORIES as EXPORT_CATEGORIES,
    build_meta,
    resolve_frame_size,
    to_coco,
    to_mot_pixels,
    to_yolo,
    to_zip_bytes,
)
from app.grok_vision import resolve_api_key
from app.pipeline import JOBS, load_job, new_job, run_job, save_job
from app.tracker import ALGORITHM_IDS, ALGORITHMS, DEFAULTS as TRACKER_DEFAULTS
from app.tracker import assign_tracks, interpolate_box, normalize_algorithm, stats

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
        "tracker": "byte-v1",
        "trackers": ["byte-v1", "ocsort-v1"],
        "tracker_ocsort": True,
        "export": "dataset-pack-v1",
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
        "tracker": job.get("tracker") or {},
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
    mixed = [{**d, "source": "detected"} for d in a.get("detections") or [] if not d.get("track_id")]
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


MAX_TRACK_FRAMES = 2000
MAX_TRACK_DETS = 600


def _clean_params(raw: Any) -> dict[str, float]:
    out: dict[str, float] = {}
    if not isinstance(raw, dict):
        return out
    for key, default in TRACKER_DEFAULTS.items():
        if key in raw:
            try:
                val = float(raw[key])
            except (TypeError, ValueError):
                raise HTTPException(400, f"param {key} must be a number")
            if key in ("track_buffer", "max_gap_fill"):
                val = max(0, min(int(val), 60))
            else:
                val = max(0.0, min(val, 1.0))
            out[key] = val
    return out


def _resolve_algorithm(payload: dict | None, params: dict | None = None) -> str:
    """Pick tracker algorithm from top-level tracker= or params.algorithm / params.tracker."""
    payload = payload or {}
    params = params or {}
    raw = (
        payload.get("tracker")
        or payload.get("algorithm")
        or params.get("tracker")
        or params.get("algorithm")
        or "byte"
    )
    algo = normalize_algorithm(raw)
    if algo not in ALGORITHMS:
        raise HTTPException(400, f"tracker must be one of {', '.join(ALGORITHMS)}")
    return algo


def _clean_frames(raw: Any) -> list[dict]:
    if not isinstance(raw, list) or not raw:
        raise HTTPException(400, "frames must be a non-empty list")
    if len(raw) > MAX_TRACK_FRAMES:
        raise HTTPException(413, f"max {MAX_TRACK_FRAMES} frames")
    frames = []
    for i, f in enumerate(raw):
        dets_in = (f or {}).get("detections") or []
        if len(dets_in) > MAX_TRACK_DETS:
            raise HTTPException(413, f"max {MAX_TRACK_DETS} detections per frame")
        dets = []
        for d in dets_in:
            try:
                dets.append({
                    "class": str(d["class"]).strip().lower(),
                    "x": float(d["x"]), "y": float(d["y"]),
                    "w": max(1e-4, float(d["w"])), "h": max(1e-4, float(d["h"])),
                    "score": float(d.get("score", 0.6)),
                })
            except (KeyError, TypeError, ValueError):
                raise HTTPException(400, f"frame {i}: each detection needs class, x, y, w, h")
        frames.append({"index": i, "t": float((f or {}).get("t", i)), "detections": dets})
    return frames



EXPORT_FORMATS = ("coco", "yolo", "mot", "zip")


def _export_payload(
    frames: list[dict],
    *,
    fmt: str,
    width: int,
    height: int,
    job_id: str | None = None,
):
    fmt = (fmt or "coco").strip().lower()
    if fmt not in EXPORT_FORMATS:
        raise HTTPException(400, f"format must be one of {', '.join(EXPORT_FORMATS)}")
    if fmt == "coco":
        return to_coco(frames, width=width, height=height, job_id=job_id)
    if fmt == "yolo":
        return {
            "files": to_yolo(frames),
            "classes": list(EXPORT_CATEGORIES),
            "meta": build_meta(frames, width=width, height=height, job_id=job_id, format_name="yolo"),
        }
    if fmt == "mot":
        return {
            "gt": to_mot_pixels(frames, width=width, height=height),
            "meta": build_meta(frames, width=width, height=height, job_id=job_id, format_name="mot"),
        }
    blob = to_zip_bytes(frames, width=width, height=height, job_id=job_id)
    name = f"grok-label-{job_id or 'adhoc'}-dataset.zip"
    return Response(
        content=blob,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{name}"'},
    )


@app.get("/api/export")
async def export_info():
    return {
        "pack": "dataset-pack-v1",
        "formats": list(EXPORT_FORMATS),
        "box_convention": "normalized_0_1_xywh_top_left",
        "categories": list(EXPORT_CATEGORIES),
        "default_size": {"width": 960, "height": 540},
        "notes": (
            "COCO/YOLO include untracked detections; MOT skips track_id is None. "
            "Interpolated gap-fill boxes are included. Default tracker is byte-v1; pass tracker=ocsort for observation-centric recovery (ocsort-v1)."
        ),
        "endpoints": {
            "POST /api/export": (
                "{frames:[{t,detections:[{class,x,y,w,h,score,track_id?}]}], "
                "format:coco|yolo|mot|zip, width?, height?} — tracks ad-hoc frames "
                "with BYTE/OC-SORT when track_ids are missing, then exports"
            ),
            "GET /api/jobs/{id}/export?format=coco|yolo|mot|zip": (
                "download labels for a finished job (zip is the dataset pack)"
            ),
        },
        "inspired_by": [
            "https://github.com/amanharshx/YOLO-Ndjson-Zip (MIT, multi-format zip idea)",
            "COCO instances JSON (public spec)",
            "YOLO / Ultralytics txt (public convention)",
            "MOT Challenge gt.txt (public format)",
        ],
    }


@app.post("/api/export")
async def export_adhoc(payload: dict = Body(...)):
    frames = _clean_frames(payload.get("frames"))
    # Preserve any caller-supplied track_id / source that survived cleaning is N/A
    # (_clean_frames strips them). Run BYTE/OC-SORT so MOT/COCO share stable ids.
    params = _clean_params(payload.get("params"))
    assign_tracks(frames, algorithm=_resolve_algorithm(payload, payload.get("params") or {}), **params)
    width = payload.get("width")
    height = payload.get("height")
    w, h = resolve_frame_size(None, width=int(width) if width else None, height=int(height) if height else None)
    fmt = str(payload.get("format") or "coco")
    return _export_payload(frames, fmt=fmt, width=w, height=h)


@app.get("/api/jobs/{job_id}/export")
async def export_job(
    job_id: str,
    format: str = Query("coco", alias="format"),
    width: int | None = None,
    height: int | None = None,
):
    job = load_job(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    if job.get("status") != "done" or not job.get("frames"):
        raise HTTPException(409, "job is not finished yet")
    frames = job["frames"]
    frames_dir = JOBS / job_id / "frames"
    w, h = resolve_frame_size(
        frames_dir if frames_dir.is_dir() else None,
        width=width,
        height=height,
    )
    return _export_payload(frames, fmt=format, width=w, height=h, job_id=job_id)


@app.get("/api/tracker")
async def tracker_info():
    return {
        "default": "byte-v1",
        "trackers": list(ALGORITHM_IDS.values()),
        "algorithms": {
            "byte": {
                "id": "byte-v1",
                "label": "BYTE two-stage association (ByteTrack-style) + motion prediction + gap fill",
                "inspired_by": "https://github.com/FoundationVision/ByteTrack (MIT)",
            },
            "ocsort": {
                "id": "ocsort-v1",
                "label": "OC-SORT observation-centric recovery + BYTE two-stage + gap fill",
                "inspired_by": "https://github.com/noahcao/OC_SORT (MIT, idea only)",
            },
        },
        "defaults": TRACKER_DEFAULTS,
        "endpoints": {
            "POST /api/track": (
                "track your own detections: {frames:[{t, detections:[{class,x,y,w,h,score}]}], "
                "tracker?:byte|ocsort, params:{}}"
            ),
            "POST /api/jobs/{id}/retrack": (
                "re-run tracking on a finished job with new params / tracker=ocsort "
                "(no new Grok calls)"
            ),
        },
    }


@app.post("/api/track")
async def track_detections(payload: dict = Body(...)):
    frames = _clean_frames(payload.get("frames"))
    params = _clean_params(payload.get("params"))
    result = assign_tracks(
        frames,
        algorithm=_resolve_algorithm(payload, payload.get("params") or {}),
        **params,
    )
    return {
        "frames": frames,
        "tracks": result["tracks"],
        "stats": stats(frames, result["tracks"]),
        "tracker": result["tracker"],
    }


@app.post("/api/jobs/{job_id}/retrack")
async def retrack(job_id: str, payload: dict = Body(default={})):
    job = load_job(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    if job.get("status") != "done" or not job.get("frames"):
        raise HTTPException(409, "job is not finished yet")
    frames = job["frames"]
    payload = payload or {}
    params = _clean_params(payload.get("params"))
    result = assign_tracks(
        frames,
        algorithm=_resolve_algorithm(payload, payload.get("params") or {}),
        **params,
    )
    job["frames"] = frames
    job["tracks"] = result["tracks"]
    job["class_counts"] = result["class_counts"]
    job["stats"] = stats(frames, result["tracks"])
    job["tracker"] = result["tracker"]
    save_job(job)
    return public_job(job)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="127.0.0.1", port=int(os.environ.get("PORT", "8787")), reload=True)
