"""Multi-object tracking for Grok Label.

The default association follows the BYTE idea from ByteTrack
(FoundationVision/ByteTrack, MIT license, https://github.com/FoundationVision/ByteTrack):

1. Keep *every* detection box and split them into high-score and low-score sets.
2. First association: high-score boxes vs. all live tracks (active + recently lost),
   using IoU against each track's motion-predicted box.
3. Second association: low-score boxes vs. the still-unmatched *active* tracks,
   IoU only. Low-score boxes are usually the occluded / blurred ones, so this
   rescues tracks through overlaps instead of spawning new IDs.
4. Tracks that miss a frame go to a "lost" pool and can be re-found for
   ``track_buffer`` sampled frames before they are retired.
5. Only unmatched high-score boxes start new tracks.

Optional algorithm ``ocsort`` (id ``ocsort-v1``) adds Observation-Centric
recovery inspired by OC-SORT (noahcao/OC_SORT, MIT license,
https://github.com/noahcao/OC_SORT) — idea only; this file is an original
pure-Python re-implementation (no numpy / lap / Kalman dependencies), tuned
for sparse ~4 fps frames:

* OCM-style virtual box: when a track is lost, also score detections against a
  short linear extrapolation from the last *observations* (not only the
  alpha-beta predicted box).
* OCR-style re-link: prefer re-linking a briefly lost track using the last
  observation box when the Kalman/predicted box has drifted (common when an
  object stops or turns behind an occluder). On recovery, velocity is
  re-seeded from last observation → new observation.

Motion prediction remains a constant-velocity alpha-beta filter on the box
centre and size. After tracking, short gaps inside a track are filled with
interpolated boxes so the Inspect player keeps a steady ID through occlusion.
"""
from __future__ import annotations

from typing import Any

CLASSES = ("person", "cardboard box", "plastic tote", "cart")
CLASS_PREFIX = {
    "person": "PERSON",
    "cardboard box": "BOX",
    "plastic tote": "TOTE",
    "cart": "CART",
}

ALGORITHMS = ("byte", "ocsort")
ALGORITHM_IDS = {
    "byte": "byte-v1",
    "ocsort": "ocsort-v1",
}

DEFAULTS: dict[str, float] = {
    "high_thresh": 0.35,    # score >= this: first association + may start a track
    "low_thresh": 0.1,      # low_thresh <= score < high_thresh: rescue-only boxes
    "match_iou": 0.2,       # min IoU (vs predicted box) in the first association
    "rescue_iou": 0.35,     # min IoU in the low-score second association
    "track_buffer": 8,      # sampled frames a lost track can be re-found (~2 s at 4 fps)
    "max_gap_fill": 8,      # longest gap (frames) filled with interpolated boxes
    "nms_iou": 0.55,
}


def normalize_algorithm(raw: Any) -> str:
    """Map API aliases to ``byte`` or ``ocsort`` (default byte)."""
    if raw is None:
        return "byte"
    name = str(raw).strip().lower().replace("_", "").replace("-", "")
    if name in ("byte", "bytev1", "bytetrack", "bytetracker"):
        return "byte"
    if name in ("ocsort", "ocsortv1", "oc"):
        return "ocsort"
    return "byte"


def iou(a: dict, b: dict) -> float:
    ax2, ay2 = a["x"] + a["w"], a["y"] + a["h"]
    bx2, by2 = b["x"] + b["w"], b["y"] + b["h"]
    ix1, iy1 = max(a["x"], b["x"]), max(a["y"], b["y"])
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = a["w"] * a["h"] + b["w"] * b["h"] - inter
    return inter / union if union else 0.0


def nms(dets: list[dict], thresh: float = 0.55) -> list[dict]:
    kept: list[dict] = []
    ordered = sorted(dets, key=lambda d: float(d.get("score") or 0), reverse=True)
    for det in ordered:
        if any(d["class"] == det["class"] and iou(d, det) >= thresh for d in kept):
            continue
        kept.append(det)
    return kept


def interpolate_box(a: dict, b: dict, alpha: float) -> dict:
    return {
        "class": a["class"],
        "track_id": a["track_id"],
        "x": a["x"] + (b["x"] - a["x"]) * alpha,
        "y": a["y"] + (b["y"] - a["y"]) * alpha,
        "w": a["w"] + (b["w"] - a["w"]) * alpha,
        "h": a["h"] + (b["h"] - a["h"]) * alpha,
        "score": min(float(a.get("score") or 0), float(b.get("score") or 0)),
        "source": "interpolated",
    }


class _Track:
    """Box state with a constant-velocity alpha-beta filter (cx, cy, w, h)."""

    ALPHA = 0.7  # position correction gain
    BETA = 0.35  # velocity correction gain

    def __init__(self, det: dict, track_id: str, frame_idx: int) -> None:
        self.track_id = track_id
        self.cls = det["class"]
        self.state = [det["x"] + det["w"] / 2, det["y"] + det["h"] / 2, det["w"], det["h"]]
        self.vel = [0.0, 0.0, 0.0, 0.0]
        self.last_frame = frame_idx
        self.hits = 1
        self.lost = False
        # Observation history for OC-SORT-style recovery (last two boxes + frames).
        self.last_obs: dict = {
            "x": det["x"], "y": det["y"], "w": det["w"], "h": det["h"], "class": det["class"]
        }
        self.prev_obs: dict | None = None
        self.prev_obs_frame: int | None = None
        self.last_obs_frame = frame_idx

    def predict(self, frame_idx: int) -> dict:
        dt = max(1, frame_idx - self.last_frame)
        cx, cy, w, h = (self.state[i] + self.vel[i] * dt for i in range(4))
        w, h = max(w, 1e-4), max(h, 1e-4)
        return {"class": self.cls, "x": cx - w / 2, "y": cy - h / 2, "w": w, "h": h}

    def obs_centric_virtual(self, frame_idx: int) -> dict | None:
        """Linear extrapolation from the last two *observations* (OCM-style).

        Falls back to the last observation alone when only one obs exists.
        """
        if self.prev_obs is None or self.prev_obs_frame is None:
            return dict(self.last_obs)
        span = max(1, self.last_obs_frame - self.prev_obs_frame)
        ahead = frame_idx - self.last_obs_frame
        # Cap how far we project so a long gap does not invent a far-away box.
        ahead = min(ahead, span * 2)
        alpha = ahead / span
        a, b = self.prev_obs, self.last_obs
        return {
            "class": self.cls,
            "x": b["x"] + (b["x"] - a["x"]) * alpha,
            "y": b["y"] + (b["y"] - a["y"]) * alpha,
            "w": max(1e-4, b["w"] + (b["w"] - a["w"]) * alpha),
            "h": max(1e-4, b["h"] + (b["h"] - a["h"]) * alpha),
        }

    def update(self, det: dict, frame_idx: int, *, obs_centric: bool = False) -> None:
        dt = max(1, frame_idx - self.last_frame)
        meas = [det["x"] + det["w"] / 2, det["y"] + det["h"] / 2, det["w"], det["h"]]

        # OCR: when recovering a lost track, re-seed velocity from last observation
        # → new observation so a drifted filter does not keep pulling the ID away.
        if obs_centric and self.lost and self.last_obs is not None:
            lo = self.last_obs
            lo_cx = lo["x"] + lo["w"] / 2
            lo_cy = lo["y"] + lo["h"] / 2
            obs_dt = max(1, frame_idx - self.last_obs_frame)
            self.vel = [
                (meas[0] - lo_cx) / obs_dt,
                (meas[1] - lo_cy) / obs_dt,
                (meas[2] - lo["w"]) / obs_dt,
                (meas[3] - lo["h"]) / obs_dt,
            ]
            self.state = meas
            self.prev_obs = dict(self.last_obs)
            self.prev_obs_frame = self.last_obs_frame
            self.last_obs = {
                "x": det["x"], "y": det["y"], "w": det["w"], "h": det["h"], "class": det["class"]
            }
            self.last_obs_frame = frame_idx
            self.last_frame = frame_idx
            self.hits += 1
            self.lost = False
            return

        if self.hits == 1:
            # Second observation: initialise velocity from the two boxes directly.
            self.vel = [(meas[i] - self.state[i]) / dt for i in range(4)]
            self.state = meas
            self.prev_obs = dict(self.last_obs)
            self.prev_obs_frame = self.last_obs_frame
            self.last_obs = {
                "x": det["x"], "y": det["y"], "w": det["w"], "h": det["h"], "class": det["class"]
            }
            self.last_obs_frame = frame_idx
            self.last_frame = frame_idx
            self.hits += 1
            self.lost = False
            return
        for i in range(4):
            pred = self.state[i] + self.vel[i] * dt
            resid = meas[i] - pred
            self.state[i] = pred + self.ALPHA * resid
            self.vel[i] = self.vel[i] + self.BETA * resid / dt
        self.prev_obs = dict(self.last_obs)
        self.prev_obs_frame = self.last_obs_frame
        self.last_obs = {
            "x": det["x"], "y": det["y"], "w": det["w"], "h": det["h"], "class": det["class"]
        }
        self.last_obs_frame = frame_idx
        self.last_frame = frame_idx
        self.hits += 1
        self.lost = False


def _similarity(pred: dict, det: dict, min_iou: float) -> float:
    """IoU, with a centre-distance fallback for boxes that jumped past each other.

    Sparse 4 fps sampling means a fast cart or a walking person can move more
    than its own width between labelled frames, so IoU drops to ~0. When IoU is
    below the gate we still allow a match if the centres are within 0.75x the
    larger box side and the sizes agree, scored strictly below any IoU match.
    """
    score = iou(pred, det)
    if score >= min_iou:
        return score
    pcx, pcy = pred["x"] + pred["w"] / 2, pred["y"] + pred["h"] / 2
    dcx, dcy = det["x"] + det["w"] / 2, det["y"] + det["h"] / 2
    dist = ((pcx - dcx) ** 2 + (pcy - dcy) ** 2) ** 0.5
    gate = 0.75 * max(pred["w"], pred["h"], det["w"], det["h"])
    area_p, area_d = pred["w"] * pred["h"], det["w"] * det["h"]
    ratio = min(area_p, area_d) / max(area_p, area_d) if max(area_p, area_d) else 0
    if dist < gate and ratio >= 0.5:
        return min_iou * 0.99 * (1 - dist / gate)
    return 0.0


def _oc_score(track: _Track, det: dict, frame_idx: int, min_iou: float) -> float:
    """Best of predicted / last-observation / OCM-virtual IoU (plus BYTE distance)."""
    pred = track.predict(frame_idx)
    best = _similarity(pred, det, min_iou)
    if track.last_obs is not None:
        best = max(best, iou(track.last_obs, det))
        best = max(best, _similarity(track.last_obs, det, min_iou))
    virtual = track.obs_centric_virtual(frame_idx)
    if virtual is not None:
        best = max(best, iou(virtual, det))
        best = max(best, _similarity(virtual, det, min_iou))
    return best


def _greedy_match(
    tracks: list[_Track],
    dets: list[dict],
    frame_idx: int,
    min_iou: float,
    allow_distance: bool = False,
    *,
    obs_centric: bool = False,
) -> tuple[list[tuple[_Track, dict]], list[_Track], list[dict]]:
    """Global greedy assignment by descending score (class-aware)."""
    pairs: list[tuple[float, int, int]] = []
    preds = [t.predict(frame_idx) for t in tracks]
    for ti, (track, pred) in enumerate(zip(tracks, preds)):
        for di, det in enumerate(dets):
            if det["class"] != track.cls:
                continue
            if obs_centric and track.lost:
                score = _oc_score(track, det, frame_idx, min_iou)
            elif allow_distance:
                score = _similarity(pred, det, min_iou)
            else:
                score = iou(pred, det)
            if score >= min_iou or (allow_distance and score > 0) or (
                obs_centric and track.lost and score >= min_iou
            ):
                pairs.append((score, ti, di))
    pairs.sort(reverse=True)
    used_t: set[int] = set()
    used_d: set[int] = set()
    matches: list[tuple[_Track, dict]] = []
    for _score, ti, di in pairs:
        if ti in used_t or di in used_d:
            continue
        used_t.add(ti)
        used_d.add(di)
        matches.append((tracks[ti], dets[di]))
    rest_t = [t for i, t in enumerate(tracks) if i not in used_t]
    rest_d = [d for i, d in enumerate(dets) if i not in used_d]
    return matches, rest_t, rest_d


def _fill_gaps(frames: list[dict[str, Any]], max_gap: int) -> int:
    """Insert interpolated boxes where a track was missed for <= max_gap frames."""
    last_seen: dict[str, tuple[int, dict]] = {}
    fills = 0
    for fi, frame in enumerate(frames):
        for det in frame.get("detections") or []:
            tid = det.get("track_id")
            if not tid or det.get("source") == "interpolated":
                continue
            prev = last_seen.get(tid)
            if prev is not None:
                pi, pdet = prev
                gap = fi - pi - 1
                if 0 < gap <= max_gap:
                    for k in range(1, gap + 1):
                        box = interpolate_box(pdet, det, k / (gap + 1))
                        frames[pi + k].setdefault("detections", []).append(box)
                        fills += 1
            last_seen[tid] = (fi, det)
    return fills


def assign_tracks(
    frames: list[dict[str, Any]],
    iou_thresh: float | None = None,
    algorithm: str | None = None,
    **params: Any,
) -> dict[str, Any]:
    """Assign stable track IDs to per-frame detections (mutates ``frames``).

    Returns track list, class counts and tracker metrics. ``iou_thresh`` is kept
    for backwards compatibility and maps to ``match_iou``.

    ``algorithm`` may be ``byte`` (default) or ``ocsort``. Callers may also pass
    ``tracker=ocsort`` / ``params.algorithm`` via the HTTP layer.
    """
    # Accept tracker= as an alias when callers spread a params dict.
    if algorithm is None and "tracker" in params:
        algorithm = params.pop("tracker")
    algo = normalize_algorithm(algorithm)
    obs_centric = algo == "ocsort"

    cfg = {**DEFAULTS, **{k: v for k, v in params.items() if k in DEFAULTS and v is not None}}
    if iou_thresh is not None:
        cfg["match_iou"] = float(iou_thresh)
    high, low = float(cfg["high_thresh"]), float(cfg["low_thresh"])
    buffer_frames = int(cfg["track_buffer"])

    counters = {name: 0 for name in CLASSES}
    active: list[_Track] = []
    lost: list[_Track] = []
    rescued_low = 0
    refound = 0
    ocm_refound = 0

    for fi, frame in enumerate(frames):
        dets = nms(
            [d for d in (frame.get("detections") or []) if d.get("source") != "interpolated"],
            float(cfg["nms_iou"]),
        )
        for det in dets:
            det["source"] = "detected"
            det.pop("track_id", None)
        high_dets = [d for d in dets if float(d.get("score") or 0) >= high]
        low_dets = [d for d in dets if low <= float(d.get("score") or 0) < high]
        dropped = [d for d in dets if float(d.get("score") or 0) < low]

        # Stage 1: confident boxes vs every live track (active + lost).
        pool = active + lost
        m1, rest_tracks, rest_high = _greedy_match(
            pool,
            high_dets,
            fi,
            float(cfg["match_iou"]),
            allow_distance=True,
            obs_centric=obs_centric,
        )
        for track, det in m1:
            was_lost = track.lost
            if was_lost:
                refound += 1
                if obs_centric:
                    # Count when last-obs / virtual beat a pure predict miss.
                    pred_only = _similarity(track.predict(fi), det, float(cfg["match_iou"]))
                    if pred_only < float(cfg["match_iou"]) and _oc_score(
                        track, det, fi, float(cfg["match_iou"])
                    ) >= float(cfg["match_iou"]):
                        ocm_refound += 1
            track.update(det, fi, obs_centric=obs_centric)
            det["track_id"] = track.track_id

        # Stage 2: low-score boxes rescue still-active tracks (IoU only).
        rest_active = [t for t in rest_tracks if not t.lost]
        m2, _unmatched_active, rest_low = _greedy_match(
            rest_active, low_dets, fi, float(cfg["rescue_iou"]), obs_centric=False
        )
        for track, det in m2:
            track.update(det, fi, obs_centric=obs_centric)
            det["track_id"] = track.track_id
            rescued_low += 1

        matched_ids = {t.track_id for t, _ in m1} | {t.track_id for t, _ in m2}

        # Stage 3: new tracks from unmatched confident boxes only.
        new_tracks: list[_Track] = []
        for det in rest_high:
            cls = det["class"]
            counters[cls] = counters.get(cls, 0) + 1
            prefix = CLASS_PREFIX.get(cls, "OBJ")
            tid = f"{prefix} T{counters[cls]:04d}"
            det["track_id"] = tid
            new_tracks.append(_Track(det, tid, fi))

        # Unmatched low boxes are background for tracking but we keep them as
        # annotations (no ID) so nothing Grok saw is silently discarded.
        for det in rest_low + dropped:
            det["track_id"] = None

        next_active: list[_Track] = []
        next_lost: list[_Track] = []
        for track in pool:
            if track.track_id in matched_ids:
                next_active.append(track)
            elif fi - track.last_frame <= buffer_frames:
                track.lost = True
                next_lost.append(track)
        active = next_active + new_tracks
        lost = next_lost
        frame["detections"] = dets

    filled = _fill_gaps(frames, int(cfg["max_gap_fill"]))

    unique: dict[str, dict] = {}
    for frame in frames:
        for det in frame.get("detections") or []:
            tid = det.get("track_id")
            if not tid:
                continue
            row = unique.setdefault(
                tid, {"track_id": tid, "class": det["class"], "frames": 0, "filled": 0}
            )
            if det.get("source") == "interpolated":
                row["filled"] += 1
            else:
                row["frames"] += 1
    tracks = list(unique.values())
    if algo == "ocsort":
        algo_label = (
            "OC-SORT observation-centric association (OC-SORT-inspired) "
            "+ BYTE two-stage + gap fill"
        )
    else:
        algo_label = "BYTE two-stage association (ByteTrack-style) + gap fill"
    return {
        "tracks": tracks,
        "class_counts": dict(counters),
        "unique_tracks": len(unique),
        "tracker": {
            "algorithm": algo_label,
            "id": ALGORITHM_IDS[algo],
            "params": cfg,
            "refound_after_miss": refound,
            "obs_centric_refound": ocm_refound,
            "rescued_low_score": rescued_low,
            "gap_filled_boxes": filled,
            "single_frame_tracks": sum(1 for t in tracks if t["frames"] <= 1),
        },
    }


def stats(frames: list[dict], tracks: list[dict]) -> dict[str, int]:
    by_class = {name: 0 for name in CLASSES}
    for track in tracks:
        cls = track.get("class")
        if cls in by_class:
            by_class[cls] += 1
    detected = sum(
        1 for f in frames for d in (f.get("detections") or []) if d.get("source") != "interpolated"
    )
    filled = sum(
        1 for f in frames for d in (f.get("detections") or []) if d.get("source") == "interpolated"
    )
    return {
        "all": sum(by_class.values()),
        "carts": by_class["cart"],
        "parcels": by_class["cardboard box"] + by_class["plastic tote"],
        "employees": by_class["person"],
        "annotations": detected,
        "gap_filled": filled,
        "classes": by_class,
    }
