"""Job guard (job-guard-v1) tests — backend only."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import app.main as main
import app.pipeline as pipeline
from app import job_guard


@pytest.fixture()
def client(tmp_path, monkeypatch):
    jobs = tmp_path / "jobs"
    jobs.mkdir()
    monkeypatch.setattr(main, "JOBS", jobs)
    monkeypatch.setattr(pipeline, "JOBS", jobs)
    calls = []

    async def fake_run(job_id):
        calls.append(job_id)

    monkeypatch.setattr(main, "run_job_safe", fake_run)
    for k in list(job_guard.COUNTERS):
        job_guard.COUNTERS[k] = 0
    c = TestClient(main.app)
    c.calls = calls
    c.jobs = jobs
    return c


def _clip(path: Path, seconds: float = 1.0) -> Path:
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", f"testsrc=size=160x120:rate=8:duration={seconds}",
         "-pix_fmt", "yuv420p", str(path)],
        check=True,
    )
    return path


def test_health_marker(client):
    body = client.get("/api/health").json()
    assert body["job_guard"] == "job-guard-v1"
    assert body["tracker"] == "byte-v1"


def test_rejects_non_video(client):
    r = client.post("/api/jobs", files={"file": ("notes.mp4", b"hello not a video", "video/mp4")})
    assert r.status_code == 400
    assert "video" in r.json()["detail"]
    assert client.calls == []
    assert not list(client.jobs.glob("upload-*"))


def test_rejects_too_large(client, monkeypatch):
    monkeypatch.setenv("MAX_UPLOAD_MB", "0.001")
    r = client.post("/api/jobs", files={"file": ("big.mp4", b"x" * 5000, "video/mp4")})
    assert r.status_code == 413
    assert job_guard.COUNTERS["blocked_too_large"] == 1


def test_rejects_too_long(client, tmp_path, monkeypatch):
    monkeypatch.setenv("MAX_CLIP_SECONDS", "1")
    clip = _clip(tmp_path / "long.mp4", 3)
    r = client.post("/api/jobs", files={"file": ("long.mp4", clip.read_bytes(), "video/mp4")})
    assert r.status_code == 400
    assert "limit" in r.json()["detail"]


def test_accepts_and_reuses_identical_clip(client, tmp_path):
    clip = _clip(tmp_path / "a.mp4", 1).read_bytes()
    r1 = client.post("/api/jobs", files={"file": ("a.mp4", clip, "video/mp4")})
    assert r1.status_code == 200, r1.text
    jid = r1.json()["job_id"]
    assert client.calls == [jid]
    job = pipeline.load_job(jid)
    assert job["fingerprint"] and job["guard"]["estimated_grok_calls"] == 4 * 5  # 1s at SAMPLE_FPS=4, grid 2 -> 1 + 4 calls per frame
    # Not finished yet -> a second upload is a new job (no reuse of unfinished work).
    # Finish the first one, then the same bytes should reuse it.
    job["status"] = "done"
    job["frames"] = [{"index": 0, "t": 0.0, "name": "frame_0001.jpg", "detections": []}]
    pipeline.save_job(job)
    r2 = client.post("/api/jobs", files={"file": ("renamed.mp4", clip, "video/mp4")})
    assert r2.status_code == 200
    body = r2.json()
    assert body["reused"] is True and body["job_id"] == jid
    assert body["job"]["status"] == "done"
    assert client.calls == [jid]
    assert job_guard.COUNTERS["reused"] == 1
    assert not list(client.jobs.glob("upload-*"))


def test_busy_limit(client, tmp_path, monkeypatch):
    monkeypatch.setenv("MAX_ACTIVE_JOBS", "1")
    a = _clip(tmp_path / "a.mp4", 1).read_bytes()
    b = _clip(tmp_path / "b.mp4", 2).read_bytes()
    assert client.post("/api/jobs", files={"file": ("a.mp4", a, "video/mp4")}).status_code == 200
    r = client.post("/api/jobs", files={"file": ("b.mp4", b, "video/mp4")})
    assert r.status_code == 429


def test_kill_switch(client, monkeypatch):
    monkeypatch.setenv("JOB_GUARD", "0")
    r = client.post("/api/jobs", files={"file": ("x.mp4", b"not video", "video/mp4")})
    assert r.status_code == 200  # legacy behaviour: queued, pipeline decides
    assert len(client.calls) == 1


def test_summary(client):
    body = client.get("/api/job-guard/summary").json()
    assert body["id"] == "job-guard-v1" and body["enabled"] is True
    assert body["limits"]["max_active_jobs"] == 2


def test_unknown_job_does_not_create_folder(client):
    assert client.get("/api/jobs/doesnotexist").status_code == 404
    assert not (client.jobs / "doesnotexist").exists()


def test_crash_marks_error(tmp_path, monkeypatch):
    import asyncio

    jobs = tmp_path / "jobs"
    jobs.mkdir()
    monkeypatch.setattr(pipeline, "JOBS", jobs)
    src = tmp_path / "bad.mp4"
    src.write_bytes(b"garbage")
    job = pipeline.new_job(src, sample_fps=4, grid=1, model="m")
    asyncio.run(pipeline.run_job_safe(job["id"]))
    after = pipeline.load_job(job["id"])
    assert after["status"] == "error"
    assert after["log"][-1]["kind"] == "error"
