"""Grok call budget (grok-call-budget-v1): a daily ceiling on paid Grok vision calls.

Backend only. job-guard-v1 caps clip size/length and concurrent jobs, but nothing
capped how many jobs (and so how many Grok calls) could run back to back in a day:
one 180 s clip at 4 fps with a 2x2 tile pass is ~3,600 calls, and anyone who can
reach ``POST /api/jobs`` could queue one after another all day. This module adds:

1. **Pre-flight reservation** (before a job is queued): the job's estimated call
   count (frames x (1 + grid^2), from job-guard's ffprobe duration) is reserved
   against
   - a per-client hourly job cap      ``GROK_CLIENT_JOBS_PER_HOUR``  (default 6)
   - a per-client daily call budget   ``GROK_CLIENT_DAILY_CALLS``    (default 4000)
   - a global daily call budget       ``GROK_DAILY_CALL_BUDGET``     (default 8000)
   If any would be exceeded the upload gets ``429`` + ``Retry-After`` and no job,
   no frames, no Grok calls are made. Nothing is reserved on rejection.
2. **Reconcile**: when a job ends (done or error) the unused part of its
   reservation is refunded, so a clip that crashes early doesn't eat the day.
3. **Runtime backstop**: every real Grok call is metered; once actual calls today
   reach ``GROK_DAILY_CALL_BUDGET x GROK_CALL_HARD_PCT/100`` (default 110 %) further
   calls raise ``GrokBudgetExhausted`` and the job is marked ``error`` (job-guard's
   crash capture) instead of spending more.

Days are UTC calendar days. State is a small JSON ledger in the jobs data dir
(survives process restarts on the same disk; resets on a fresh deploy because the
service has no volume). Client keys are stored as short SHA-256 hashes, not IPs.
Re-used jobs (identical clip) and ``/retrack`` make no Grok calls and cost nothing.

Kill switch: ``GROK_CALL_BUDGET=0`` turns off reservation and the backstop
(metering/counters keep running so the summary stays honest).

Ideas, not code, from: two-phase reserve -> reconcile LLM budgets
(Elijas/token-throttle, jackChallis/tokengate, pradyumnvermaa/hardcap — all
Apache-2.0, idea only), "check before, record after" pre-call budget enforcement
(r/ArtificialInteligence, r/FinOps threads), OpenAI cookbook per-run spending
controller, and Retry-After style 429s from laurentS/slowapi (MIT) and
animir/node-rate-limiter-flexible (ISC). Original code; nothing vendored.
"""
from __future__ import annotations

import contextvars
import datetime as _dt
import hashlib
import json
import os
import threading
import time
from pathlib import Path
from typing import Any

GROK_CALL_BUDGET_ID = "grok-call-budget-v1"
LEDGER_NAME = "_grok_call_budget.json"

_LOCK = threading.RLock()
_LEDGER_DIR: Path | None = None
_STATE: dict[str, Any] | None = None

# job id of the pipeline run currently making calls (set in pipeline.run_job).
CURRENT_JOB: contextvars.ContextVar[str | None] = contextvars.ContextVar("grok_budget_job", default=None)

COUNTERS: dict[str, int] = {
    "reserved_jobs": 0,
    "blocked_client_jobs_per_hour": 0,
    "blocked_client_daily": 0,
    "blocked_global_daily": 0,
    "backstop_blocked_calls": 0,
    "refunded_calls": 0,
    "metered_calls": 0,
}


class BudgetReject(Exception):
    def __init__(self, status: int, reason: str, detail: str, retry_after: int):
        super().__init__(detail)
        self.status = status
        self.reason = reason
        self.detail = detail
        self.retry_after = max(1, int(retry_after))


class GrokBudgetExhausted(RuntimeError):
    """Raised before a Grok call when today's hard ceiling is already reached."""


def _env_int(name: str, default: int) -> int:
    try:
        return int(float(os.environ.get(name) or default))
    except (TypeError, ValueError):
        return default


def enabled() -> bool:
    return str(os.environ.get("GROK_CALL_BUDGET", "1")).strip().lower() not in {"0", "false", "off", "no"}


def limits() -> dict[str, int]:
    daily = max(0, _env_int("GROK_DAILY_CALL_BUDGET", 8000))
    pct = max(100, _env_int("GROK_CALL_HARD_PCT", 110))
    return {
        "daily_calls": daily,
        "client_daily_calls": max(0, _env_int("GROK_CLIENT_DAILY_CALLS", 4000)),
        "client_jobs_per_hour": max(0, _env_int("GROK_CLIENT_JOBS_PER_HOUR", 6)),
        "hard_ceiling_pct": pct,
        "hard_ceiling_calls": (daily * pct) // 100,
    }


def _today(now: float | None = None) -> str:
    return _dt.datetime.fromtimestamp(now if now is not None else time.time(), _dt.timezone.utc).strftime("%Y-%m-%d")


def _seconds_to_reset(now: float | None = None) -> int:
    now = now if now is not None else time.time()
    d = _dt.datetime.fromtimestamp(now, _dt.timezone.utc)
    nxt = (d + _dt.timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return max(1, int((nxt - d).total_seconds()))


def _resets_at(now: float | None = None) -> str:
    now = now if now is not None else time.time()
    d = _dt.datetime.fromtimestamp(now, _dt.timezone.utc)
    nxt = (d + _dt.timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return nxt.isoformat().replace("+00:00", "Z")


def client_hash(raw: str | None) -> str:
    return hashlib.sha256((raw or "unknown").encode("utf-8")).hexdigest()[:16]


def client_key_from_request(request: Any) -> str:
    """Best-effort client id behind Railway's proxy, hashed (no raw IPs kept)."""
    headers = getattr(request, "headers", {}) or {}
    raw = (headers.get("x-real-ip") or "").strip()
    if not raw:
        xff = (headers.get("x-forwarded-for") or "").strip()
        if xff:
            # The proxy appends the address it saw; the right-most hop is the hardest to spoof.
            raw = xff.split(",")[-1].strip()
    if not raw:
        client = getattr(request, "client", None)
        raw = getattr(client, "host", "") or "unknown"
    return client_hash(raw)


def _fresh_state(day: str) -> dict[str, Any]:
    return {"day": day, "reserved": 0, "actual": 0, "clients": {}, "jobs": {}, "job_calls": {}, "open": {}}


def configure(ledger_dir: Path) -> None:
    """Point the ledger at a directory (the jobs root). Reloads state if it changed."""
    global _LEDGER_DIR, _STATE
    with _LOCK:
        new = Path(ledger_dir)
        if _LEDGER_DIR != new:
            _LEDGER_DIR = new
            _STATE = None


def _ledger_path() -> Path | None:
    return (_LEDGER_DIR / LEDGER_NAME) if _LEDGER_DIR else None


def _load(now: float | None = None) -> dict[str, Any]:
    global _STATE
    day = _today(now)
    if _STATE is None:
        state = None
        path = _ledger_path()
        if path and path.exists():
            try:
                state = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                state = None
        _STATE = state if isinstance(state, dict) and state.get("day") else _fresh_state(day)
        for key, default in _fresh_state(day).items():
            _STATE.setdefault(key, default)
    if _STATE.get("day") != day:
        # New UTC day: start fresh totals. Yesterday's open reservations simply expire.
        _STATE = _fresh_state(day)
    return _STATE


def _save() -> None:
    path = _ledger_path()
    if not path or _STATE is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(_STATE), encoding="utf-8")
        os.replace(tmp, path)
    except OSError:
        pass


def bump(key: str, n: int = 1) -> None:
    with _LOCK:
        COUNTERS[key] = COUNTERS.get(key, 0) + n


def reserve(client: str, estimated_calls: int | None, *, fallback_calls: int, now: float | None = None) -> dict[str, Any]:
    """Atomically check all caps and hold ``estimated_calls``. Raises BudgetReject.

    Returns a reservation dict to store on the job: {id, day, client, reserved}.
    """
    now = now if now is not None else time.time()
    est = int(estimated_calls) if estimated_calls and estimated_calls > 0 else int(max(1, fallback_calls))
    lim = limits()
    with _LOCK:
        state = _load(now)
        if not enabled():
            return {"id": GROK_CALL_BUDGET_ID, "day": state["day"], "client": client, "reserved": 0, "enforced": False}
        # 1) per-client hourly job cap
        stamps = [t for t in state["jobs"].get(client, []) if now - t < 3600]
        state["jobs"][client] = stamps
        if lim["client_jobs_per_hour"] and len(stamps) >= lim["client_jobs_per_hour"]:
            COUNTERS["blocked_client_jobs_per_hour"] += 1
            retry = int(3600 - (now - min(stamps))) + 1
            _save()
            raise BudgetReject(
                429,
                "client_jobs_per_hour",
                f"You've started {len(stamps)} clips in the last hour (limit {lim['client_jobs_per_hour']}). "
                f"Try again in about {max(1, retry // 60)} min.",
                retry,
            )
        # 2) per-client daily calls
        used_client = int((state["clients"].get(client) or {}).get("reserved", 0))
        if lim["client_daily_calls"] and used_client + est > lim["client_daily_calls"]:
            COUNTERS["blocked_client_daily"] += 1
            raise BudgetReject(
                429,
                "client_daily_calls",
                f"This clip needs about {est} Grok calls and your daily labeling budget has "
                f"{max(0, lim['client_daily_calls'] - used_client)} left. Use a shorter clip or try again after 00:00 UTC.",
                _seconds_to_reset(now),
            )
        # 3) global daily calls (reserved and actually spent, whichever is higher)
        used_global = max(int(state["reserved"]), int(state["actual"]))
        if lim["daily_calls"] and used_global + est > lim["daily_calls"]:
            COUNTERS["blocked_global_daily"] += 1
            raise BudgetReject(
                429,
                "global_daily_calls",
                "Today's Grok labeling budget is used up. Try again after 00:00 UTC.",
                _seconds_to_reset(now),
            )
        stamps.append(now)
        state["jobs"][client] = stamps
        entry = state["clients"].setdefault(client, {"reserved": 0, "jobs": 0})
        entry["reserved"] = int(entry.get("reserved", 0)) + est
        entry["jobs"] = int(entry.get("jobs", 0)) + 1
        state["reserved"] = int(state["reserved"]) + est
        COUNTERS["reserved_jobs"] += 1
        _save()
        return {"id": GROK_CALL_BUDGET_ID, "day": state["day"], "client": client, "reserved": est, "enforced": True}


def release(reservation: dict[str, Any] | None) -> None:
    """Give back a reservation whose job never started (e.g. copy failed)."""
    if not reservation or not reservation.get("reserved"):
        return
    with _LOCK:
        state = _load()
        if reservation.get("day") != state["day"]:
            return
        n = int(reservation["reserved"])
        state["reserved"] = max(0, int(state["reserved"]) - n)
        entry = state["clients"].get(reservation.get("client"))
        if entry:
            entry["reserved"] = max(0, int(entry.get("reserved", 0)) - n)
        COUNTERS["refunded_calls"] += n
        _save()


def attach_job(job_id: str, reservation: dict[str, Any] | None) -> None:
    """Remember which reservation a queued job holds (for refund on finish)."""
    if not reservation or not reservation.get("reserved"):
        return
    with _LOCK:
        state = _load()
        state.setdefault("open", {})[job_id] = {
            "day": reservation.get("day"),
            "client": reservation.get("client"),
            "reserved": int(reservation.get("reserved") or 0),
        }
        _save()


def settle(job_id: str) -> dict[str, Any]:
    """Refund the unused part of a finished/failed job's reservation. Idempotent."""
    with _LOCK:
        state = _load()
        used = int(state.get("job_calls", {}).pop(job_id, 0))
        held = state.get("open", {}).pop(job_id, None)
        if not held:
            _save()
            return {"job_id": job_id, "used": used, "refunded": 0}
        refund = max(0, int(held.get("reserved", 0)) - used)
        if refund and held.get("day") == state["day"]:
            state["reserved"] = max(0, int(state["reserved"]) - refund)
            entry = state["clients"].get(held.get("client")) or None
            if entry:
                entry["reserved"] = max(0, int(entry.get("reserved", 0)) - refund)
            COUNTERS["refunded_calls"] += refund
        _save()
        return {"job_id": job_id, "used": used, "refunded": refund if held.get("day") == state["day"] else 0}


def meter_call(now: float | None = None) -> None:
    """Call right before each paid Grok request. Raises GrokBudgetExhausted at the hard ceiling."""
    lim = limits()
    with _LOCK:
        state = _load(now)
        if enabled() and lim["daily_calls"] and int(state["actual"]) >= lim["hard_ceiling_calls"]:
            COUNTERS["backstop_blocked_calls"] += 1
            raise GrokBudgetExhausted(
                "Today's Grok call budget is used up, so labeling stopped before spending more. "
                "It resets at 00:00 UTC."
            )
        state["actual"] = int(state["actual"]) + 1
        job = CURRENT_JOB.get()
        if job:
            calls = state.setdefault("job_calls", {})
            calls[job] = int(calls.get(job, 0)) + 1
        COUNTERS["metered_calls"] += 1
        # Persist every 25 calls to keep disk writes low; reservations already persisted.
        if int(state["actual"]) % 25 == 0:
            _save()


def summary(now: float | None = None, client: str | None = None) -> dict[str, Any]:
    lim = limits()
    now_ts = now if now is not None else time.time()
    with _LOCK:
        state = _load(now)
        you = None
        if client:
            entry = state.get("clients", {}).get(client) or {}
            stamps = [t for t in state.get("jobs", {}).get(client, []) if now_ts - t < 3600]
            you = {
                "client": client,
                "reserved_calls_today": int(entry.get("reserved", 0)),
                "remaining_calls_today": max(0, lim["client_daily_calls"] - int(entry.get("reserved", 0))),
                "jobs_last_hour": len(stamps),
            }
        counters = dict(COUNTERS)
        reserved = int(state["reserved"])
        actual = int(state["actual"])
        clients = len(state.get("clients") or {})
        open_jobs = len(state.get("open") or {})
    used = max(reserved, actual)
    return {
        "id": GROK_CALL_BUDGET_ID,
        "enabled": enabled(),
        "limits": lim,
        "day_utc": state["day"],
        "resets_at": _resets_at(now),
        "today": {
            "reserved_calls": reserved,
            "actual_calls": actual,
            "remaining_calls": max(0, lim["daily_calls"] - used) if lim["daily_calls"] else None,
            "clients": clients,
            "open_reservations": open_jobs,
        },
        "you": you,
        "counters": counters,
        "notes": (
            "Pre-flight reserves each job's estimated Grok calls (frames x (1+grid^2)) against per-client and global "
            "UTC-daily budgets, refunds unused calls when the job ends, and a runtime backstop stops calls at the hard "
            "ceiling. GROK_CALL_BUDGET=0 disables enforcement. Counters are per process; the daily ledger lives on the "
            "service disk and resets on a fresh deploy (no volume)."
        ),
    }


def reset_for_tests(ledger_dir: Path | None = None) -> None:
    global _STATE, _LEDGER_DIR
    with _LOCK:
        _STATE = None
        _LEDGER_DIR = Path(ledger_dir) if ledger_dir else None
        for k in COUNTERS:
            COUNTERS[k] = 0
