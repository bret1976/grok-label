"""grok-call-budget-v1 tests — backend only."""
from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import app.main as main
import app.pipeline as pipeline
from app import grok_call_budget as gcb
from app import job_guard


@pytest.fixture()
def client(tmp_path, monkeypatch):
    jobs = tmp_path / "jobs"
    jobs.mkdir()
    monkeypatch.setattr(main, "JOBS", jobs)
    monkeypatch.setattr(pipeline, "JOBS", jobs)
    monkeypatch.setenv("SAMPLE_FPS", "4")
    monkeypatch.setenv("TILE_GRID", "2")
    calls = []

    async def fake_run(job_id):
        calls.append(job_id)

    monkeypatch.setattr(main, "run_job_safe", fake_run)
    for k in list(job_guard.COUNTERS):
        job_guard.COUNTERS[k] = 0
    monkeypatch.setenv("MAX_ACTIVE_JOBS", "100")
    gcb.reset_for_tests(jobs)
    c = TestClient(main.app)
    c.calls = calls
    c.jobs = jobs
    return c


def _clip(path: Path, seconds: float = 1.0, color: str = "testsrc") -> Path:
    src = f"{color}=size=160x120:rate=8:duration={seconds}" if color == "testsrc" else f"color=c={color}:size=160x120:rate=8:duration={seconds}"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", src, "-pix_fmt", "yuv420p", str(path)], check=True)
    return path


def _upload(client, path: Path, ip: str = "1.2.3.4"):
    return client.post(
        "/api/jobs",
        files={"file": (path.name, path.read_bytes(), "video/mp4")},
        headers={"x-real-ip": ip},
    )


def test_health_marker_and_summary(client):
    body = client.get("/api/health").json()
    assert body["grok_call_budget"] == "grok-call-budget-v1"
    assert body["grok_call_budget_enabled"] is True
    assert body["job_guard"] == "job-guard-v1"
    s = client.get("/api/grok-call-budget/summary").json()
    assert s["id"] == "grok-call-budget-v1"
    assert s["limits"]["daily_calls"] == 8000
    assert s["limits"]["hard_ceiling_calls"] == 8800
    assert s["today"]["reserved_calls"] == 0


def test_reserves_estimate_and_allows(client, tmp_path):
    clip = _clip(tmp_path / "a.mp4", 1.0)
    r = _upload(client, clip)
    assert r.status_code == 200, r.text
    s = client.get("/api/grok-call-budget/summary").json()
    # 1 s * 4 fps = 4 frames * (1 + 2*2) = 20 calls
    assert s["today"]["reserved_calls"] == 20
    assert s["today"]["open_reservations"] == 1
    job = pipeline.load_job(r.json()["job_id"])
    assert job["budget"]["reserved_calls"] == 20
    you = client.get("/api/grok-call-budget/summary", headers={"x-real-ip": "1.2.3.4"}).json()["you"]
    assert you["reserved_calls_today"] == 20 and you["jobs_last_hour"] == 1
    other = client.get("/api/grok-call-budget/summary", headers={"x-real-ip": "4.4.4.4"}).json()["you"]
    assert other["reserved_calls_today"] == 0 and other["client"] != you["client"]


def test_global_daily_budget_blocks_with_retry_after(client, tmp_path, monkeypatch):
    monkeypatch.setenv("GROK_DAILY_CALL_BUDGET", "30")
    a = _upload(client, _clip(tmp_path / "a.mp4", 1.0), ip="9.9.9.1")
    assert a.status_code == 200
    b = _upload(client, _clip(tmp_path / "b.mp4", 1.0, color="red"), ip="9.9.9.2")
    assert b.status_code == 429
    assert "budget" in b.json()["detail"].lower()
    assert int(b.headers["retry-after"]) > 0
    assert len(client.calls) == 1
    assert not list(client.jobs.glob("upload-*"))
    assert gcb.COUNTERS["blocked_global_daily"] == 1


def test_client_daily_budget_is_per_client(client, tmp_path, monkeypatch):
    monkeypatch.setenv("GROK_CLIENT_DAILY_CALLS", "25")
    assert _upload(client, _clip(tmp_path / "a.mp4", 1.0), ip="5.5.5.5").status_code == 200
    r = _upload(client, _clip(tmp_path / "b.mp4", 1.0, color="blue"), ip="5.5.5.5")
    assert r.status_code == 429
    assert "daily labeling budget" in r.json()["detail"]
    # different client still fine
    assert _upload(client, _clip(tmp_path / "c.mp4", 1.0, color="green"), ip="6.6.6.6").status_code == 200


def test_client_jobs_per_hour(client, tmp_path, monkeypatch):
    monkeypatch.setenv("GROK_CLIENT_JOBS_PER_HOUR", "2")
    colors = ["red", "blue", "green"]
    codes = [_upload(client, _clip(tmp_path / f"{c}.mp4", 0.5, color=c), ip="7.7.7.7").status_code for c in colors]
    assert codes == [200, 200, 429]
    assert gcb.COUNTERS["blocked_client_jobs_per_hour"] == 1


def test_reused_clip_costs_nothing(client, tmp_path):
    clip = _clip(tmp_path / "a.mp4", 1.0)
    r = _upload(client, clip)
    job = pipeline.load_job(r.json()["job_id"])
    job["status"] = "done"
    job["frames"] = [{"index": 0, "t": 0, "detections": []}]
    pipeline.save_job(job)
    before = client.get("/api/grok-call-budget/summary").json()["today"]["reserved_calls"]
    r2 = _upload(client, clip)
    assert r2.json().get("reused") is True
    after = client.get("/api/grok-call-budget/summary").json()["today"]["reserved_calls"]
    assert after == before


def test_kill_switch(client, tmp_path, monkeypatch):
    monkeypatch.setenv("GROK_CALL_BUDGET", "0")
    monkeypatch.setenv("GROK_DAILY_CALL_BUDGET", "1")
    assert _upload(client, _clip(tmp_path / "a.mp4", 1.0)).status_code == 200
    assert client.get("/api/health").json()["grok_call_budget_enabled"] is False
    # backstop also off
    for _ in range(3):
        gcb.meter_call()


def test_settle_refunds_unused_and_backstop(tmp_path, monkeypatch):
    gcb.reset_for_tests(tmp_path)
    monkeypatch.setenv("GROK_DAILY_CALL_BUDGET", "100")
    res = gcb.reserve("c1", 40, fallback_calls=999)
    gcb.attach_job("j1", res)
    token = gcb.CURRENT_JOB.set("j1")
    try:
        for _ in range(10):
            gcb.meter_call()
    finally:
        gcb.CURRENT_JOB.reset(token)
    out = gcb.settle("j1")
    assert out == {"job_id": "j1", "used": 10, "refunded": 30}
    s = gcb.summary()
    assert s["today"]["reserved_calls"] == 10
    assert s["today"]["actual_calls"] == 10
    assert gcb.settle("j1")["refunded"] == 0  # idempotent
    # backstop: hard ceiling = 110 calls
    for _ in range(100):
        gcb.meter_call()
    with pytest.raises(gcb.GrokBudgetExhausted):
        gcb.meter_call()
    assert gcb.COUNTERS["backstop_blocked_calls"] == 1


def test_ledger_persists_across_reload(tmp_path):
    gcb.reset_for_tests(tmp_path)
    gcb.reserve("c1", 50, fallback_calls=1)
    gcb.reset_for_tests(tmp_path)  # simulate process restart on same disk
    assert gcb.summary()["today"]["reserved_calls"] == 50


def test_pipeline_crash_settles_and_marks_error(tmp_path, monkeypatch):
    jobs = tmp_path / "jobs"
    jobs.mkdir()
    monkeypatch.setattr(pipeline, "JOBS", jobs)
    gcb.reset_for_tests(jobs)
    monkeypatch.setenv("GROK_DAILY_CALL_BUDGET", "1000")
    clip = _clip(tmp_path / "a.mp4", 1.0)
    job = pipeline.new_job(clip, sample_fps=4, grid=2, model="m")
    res = gcb.reserve("c1", 20, fallback_calls=1)
    gcb.attach_job(job["id"], res)

    async def fake_label(path, *, model, tile_note="x"):
        gcb.meter_call()
        raise RuntimeError("Grok vision error 500: boom")

    monkeypatch.setattr("app.grok_vision.label_image", fake_label)
    asyncio.run(pipeline.run_job_safe(job["id"]))
    j = pipeline.load_job(job["id"])
    assert j["status"] == "error"
    s = gcb.summary()
    assert s["today"]["open_reservations"] == 0
    assert s["today"]["reserved_calls"] == s["today"]["actual_calls"] <= 20
    assert s["today"]["actual_calls"] >= 1


def test_client_key_hashes_ip():
    class R:
        headers = {"x-forwarded-for": "6.6.6.6, 10.0.0.1"}
        client = None

    k = gcb.client_key_from_request(R())
    assert k == gcb.client_hash("10.0.0.1")
    assert "10.0.0.1" not in k
