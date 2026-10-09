"""stale-job-watch-v1 tests — backend only."""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import app.main as main
import app.pipeline as pipeline
from app import stale_job_watch as sjw


@pytest.fixture()
def client(tmp_path, monkeypatch):
    jobs = tmp_path / "jobs"
    jobs.mkdir()
    monkeypatch.setattr(main, "JOBS", jobs)
    monkeypatch.setattr(pipeline, "JOBS", jobs)
    for k in list(sjw.COUNTERS):
        sjw.COUNTERS[k] = 0
    monkeypatch.delenv("STALE_JOB_WATCH", raising=False)
    monkeypatch.delenv("STALE_JOB_WATCH_MINUTES", raising=False)
    monkeypatch.delenv("JOB_STALE_MINUTES", raising=False)
    c = TestClient(main.app)
    c.jobs = jobs
    return c


def _write_job(jobs: Path, job_id: str, status: str, *, mtime_age_s: float = 0) -> Path:
    d = jobs / job_id
    d.mkdir(parents=True, exist_ok=True)
    path = d / "job.json"
    job = {
        "id": job_id,
        "status": status,
        "stage": "annotate",
        "progress": 0.4,
        "log": [],
        "frames": [],
        "tracks": [],
    }
    path.write_text(json.dumps(job, indent=2), encoding="utf-8")
    if mtime_age_s:
        past = time.time() - mtime_age_s
        import os

        os.utime(path, (past, past))
    return path


def test_health_marker(client):
    body = client.get("/api/health").json()
    assert body["stale_job_watch"] == "stale-job-watch-v1"
    assert body["stale_job_watch_enabled"] is True
    assert body["job_guard"] == "job-guard-v1"


def test_summary_marks_stale_running(client, monkeypatch):
    monkeypatch.setenv("STALE_JOB_WATCH_MINUTES", "1")
    _write_job(client.jobs, "oldrun", "running", mtime_age_s=120)
    _write_job(client.jobs, "fresh", "running", mtime_age_s=5)
    _write_job(client.jobs, "doneok", "done", mtime_age_s=9999)

    r = client.get("/api/stale-job-watch/summary")
    assert r.status_code == 200
    body = r.json()
    assert body["id"] == "stale-job-watch-v1"
    assert body["enabled"] is True
    assert body["last_sweep_marked"] == 1
    assert body["counters"]["marked_stale"] >= 1

    old = json.loads((client.jobs / "oldrun" / "job.json").read_text())
    assert old["status"] == "error"
    assert old["error"] == "stale_timeout"
    assert old["stale"]["id"] == "stale-job-watch-v1"
    assert any("stale" in (e.get("text") or "").lower() for e in old["log"])

    fresh = json.loads((client.jobs / "fresh" / "job.json").read_text())
    assert fresh["status"] == "running"

    done = json.loads((client.jobs / "doneok" / "job.json").read_text())
    assert done["status"] == "done"


def test_kill_switch_skips_reap(client, monkeypatch):
    monkeypatch.setenv("STALE_JOB_WATCH", "0")
    monkeypatch.setenv("STALE_JOB_WATCH_MINUTES", "1")
    _write_job(client.jobs, "stuck", "queued", mtime_age_s=600)
    r = client.get("/api/stale-job-watch/summary")
    assert r.status_code == 200
    assert r.json()["enabled"] is False
    assert r.json()["last_sweep_marked"] == 0
    stuck = json.loads((client.jobs / "stuck" / "job.json").read_text())
    assert stuck["status"] == "queued"


def test_create_job_sweeps_before_busy(client, tmp_path, monkeypatch):
    """A stale 'running' job must not permanently occupy an active slot once watch is on."""
    import subprocess

    monkeypatch.setenv("STALE_JOB_WATCH_MINUTES", "1")
    monkeypatch.setenv("MAX_ACTIVE_JOBS", "1")
    _write_job(client.jobs, "ghost", "running", mtime_age_s=600)

    calls = []

    async def fake_run(job_id):
        calls.append(job_id)

    monkeypatch.setattr(main, "run_job_safe", fake_run)

    clip = tmp_path / "a.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=160x120:rate=8:duration=1",
            "-pix_fmt",
            "yuv420p",
            str(clip),
        ],
        check=True,
    )
    r = client.post("/api/jobs", files={"file": ("a.mp4", clip.read_bytes(), "video/mp4")})
    assert r.status_code == 200, r.text
    ghost = json.loads((client.jobs / "ghost" / "job.json").read_text())
    assert ghost["status"] == "error"
    assert ghost["error"] == "stale_timeout"
    assert len(calls) == 1


def test_sweep_idempotent(client, monkeypatch):
    monkeypatch.setenv("STALE_JOB_WATCH_MINUTES", "1")
    _write_job(client.jobs, "once", "running", mtime_age_s=600)
    a = sjw.sweep(client.jobs)
    b = sjw.sweep(client.jobs)
    assert len(a["marked"]) == 1
    assert len(b["marked"]) == 0
    assert sjw.COUNTERS["marked_stale"] == 1
