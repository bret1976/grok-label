"""Export / annotation QA for Grok Label dataset packs.

Original pure-Python checks for normalized 0–1 xywh boxes (top-left) and
track continuity before COCO / YOLO / MOT export. Idea inspired by
Rituparno-Majumdar/annocheck (MIT,
https://github.com/Rituparno-Majumdar/annocheck) — validate labels for
malformed boxes, size issues, and dataset quality — plus public COCO /
YOLO / MOT conventions. Nothing vendored; no new UI.
"""
from __future__ import annotations

from typing import Any

from app.tracker import CLASSES

EXPORT_QA_ID = "export-qa-v1"
KNOWN_CLASSES = frozenset(CLASSES)

# Soft thresholds (normalized space).
ZERO_AREA_EPS = 1e-8
NEAR_DUP_IOU = 0.95
DEFAULT_MAX_TRACK_GAP = 8  # frames; gaps larger than this (without fill) flag


def _iou(a: dict, b: dict) -> float:
    ax2, ay2 = a["x"] + a["w"], a["y"] + a["h"]
    bx2, by2 = b["x"] + b["w"], b["y"] + b["h"]
    ix1, iy1 = max(a["x"], b["x"]), max(a["y"], b["y"])
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = a["w"] * a["h"] + b["w"] * b["h"] - inter
    return inter / union if union else 0.0


def _frame_index(frame: dict, fallback_i: int) -> int:
    idx = frame.get("index")
    if isinstance(idx, int):
        return idx
    return fallback_i


def _issue(
    code: str,
    message: str,
    *,
    frame_index: int | None = None,
    det_index: int | None = None,
    track_id: Any = None,
    severity: str = "error",
    detail: dict | None = None,
) -> dict:
    row: dict[str, Any] = {
        "code": code,
        "severity": severity,
        "message": message,
    }
    if frame_index is not None:
        row["frame_index"] = frame_index
    if det_index is not None:
        row["det_index"] = det_index
    if track_id is not None:
        row["track_id"] = track_id
    if detail:
        row["detail"] = detail
    return row


def validate_frames(
    frames: list[dict] | None,
    *,
    max_track_gap: int = DEFAULT_MAX_TRACK_GAP,
    known_classes: frozenset[str] | None = None,
) -> dict[str, Any]:
    """Scan labeled frames and return a QA report (does not mutate frames).

    Checks (backend only):
    * missing / non-numeric box fields
    * zero or negative area
    * out-of-bounds (raw box not fully inside [0, 1] before clamp)
    * unknown / blank class
    * near-duplicate boxes in the same frame (same class, high IoU)
    * track continuity gaps (reappear after > max_track_gap missed frames)
    * fragmented tracks (multiple gap events) as a continuity hint
    """
    classes = known_classes if known_classes is not None else KNOWN_CLASSES
    issues: list[dict] = []
    counts = {
        "frames": 0,
        "detections": 0,
        "tracked": 0,
        "untracked": 0,
        "interpolated": 0,
        "errors": 0,
        "warnings": 0,
    }
    by_code: dict[str, int] = {}
    # track_id -> list of frame indices where a non-interpolated box appears
    track_frames: dict[str, list[int]] = {}

    for fi, frame in enumerate(frames or []):
        counts["frames"] += 1
        fidx = _frame_index(frame, fi)
        dets = list(frame.get("detections") or [])
        seen_for_dup: list[dict] = []

        for di, det in enumerate(dets):
            counts["detections"] += 1
            if det.get("source") == "interpolated":
                counts["interpolated"] += 1

            cls_raw = det.get("class")
            cls = str(cls_raw).strip().lower() if cls_raw is not None else ""
            if not cls:
                issues.append(
                    _issue(
                        "bad_class",
                        "blank or missing class",
                        frame_index=fidx,
                        det_index=di,
                    )
                )
            elif cls not in classes:
                issues.append(
                    _issue(
                        "bad_class",
                        f"unknown class '{cls}'",
                        frame_index=fidx,
                        det_index=di,
                        detail={"class": cls, "allowed": sorted(classes)},
                    )
                )

            box_ok = True
            try:
                x = float(det["x"])
                y = float(det["y"])
                w = float(det["w"])
                h = float(det["h"])
            except (KeyError, TypeError, ValueError):
                box_ok = False
                issues.append(
                    _issue(
                        "invalid_box",
                        "detection needs numeric x, y, w, h",
                        frame_index=fidx,
                        det_index=di,
                    )
                )

            if box_ok:
                if w <= 0 or h <= 0 or (w * h) <= ZERO_AREA_EPS:
                    issues.append(
                        _issue(
                            "zero_area",
                            "box has zero or negative area",
                            frame_index=fidx,
                            det_index=di,
                            detail={"w": w, "h": h},
                        )
                    )
                # Out of bounds before any clamp: any edge outside [0, 1].
                oob_parts: list[str] = []
                if x < 0 or y < 0:
                    oob_parts.append("origin_negative")
                if w < 0 or h < 0:
                    oob_parts.append("negative_size")
                if x + w > 1.0 + 1e-9 or y + h > 1.0 + 1e-9:
                    oob_parts.append("extends_past_1")
                if x > 1.0 or y > 1.0:
                    oob_parts.append("origin_past_1")
                if oob_parts:
                    issues.append(
                        _issue(
                            "out_of_bounds",
                            "box not fully inside normalized [0, 1]",
                            frame_index=fidx,
                            det_index=di,
                            detail={"x": x, "y": y, "w": w, "h": h, "flags": oob_parts},
                        )
                    )

                # Near-duplicate within frame (same class).
                if cls:
                    for prev in seen_for_dup:
                        if prev["class"] != cls:
                            continue
                        if _iou(prev, {"x": x, "y": y, "w": w, "h": h}) >= NEAR_DUP_IOU:
                            issues.append(
                                _issue(
                                    "duplicate_box",
                                    "near-duplicate box in same frame (same class)",
                                    frame_index=fidx,
                                    det_index=di,
                                    severity="warning",
                                    detail={"iou_gate": NEAR_DUP_IOU, "other_det_index": prev["di"]},
                                )
                            )
                            break
                    seen_for_dup.append({"class": cls, "x": x, "y": y, "w": w, "h": h, "di": di})

            tid = det.get("track_id")
            if tid is None or str(tid).strip() == "":
                counts["untracked"] += 1
            else:
                counts["tracked"] += 1
                if det.get("source") != "interpolated":
                    track_frames.setdefault(str(tid), []).append(fidx)

    # Track continuity: gaps between observed (non-interpolated) frames.
    gap_events = 0
    fragmented_tracks = 0
    for tid, idxs in track_frames.items():
        idxs_sorted = sorted(set(idxs))
        gaps_for_track = 0
        for a, b in zip(idxs_sorted, idxs_sorted[1:]):
            gap = b - a - 1
            if gap > max_track_gap:
                gap_events += 1
                gaps_for_track += 1
                issues.append(
                    _issue(
                        "track_gap",
                        f"track reappears after {gap} missed frames (max {max_track_gap})",
                        track_id=tid,
                        severity="warning",
                        detail={
                            "from_frame": a,
                            "to_frame": b,
                            "gap": gap,
                            "max_track_gap": max_track_gap,
                        },
                    )
                )
        if gaps_for_track >= 2:
            fragmented_tracks += 1
            issues.append(
                _issue(
                    "track_fragmented",
                    f"track has {gaps_for_track} large gaps (continuity risk)",
                    track_id=tid,
                    severity="warning",
                    detail={"large_gaps": gaps_for_track},
                )
            )

    for iss in issues:
        code = iss["code"]
        by_code[code] = by_code.get(code, 0) + 1
        if iss.get("severity") == "warning":
            counts["warnings"] += 1
        else:
            counts["errors"] += 1

    unique_tracks = len(track_frames)
    # Continuity score: 1.0 if no large gaps; decays with gap events / tracks.
    if unique_tracks == 0:
        continuity = 1.0 if counts["detections"] == 0 else 0.5
    else:
        continuity = max(0.0, 1.0 - (gap_events / max(unique_tracks, 1)))

    ok = counts["errors"] == 0
    return {
        "ok": ok,
        "pack": EXPORT_QA_ID,
        "summary": {
            **counts,
            "unique_tracks": unique_tracks,
            "track_gap_events": gap_events,
            "fragmented_tracks": fragmented_tracks,
            "continuity_score": round(continuity, 4),
            "issue_count": len(issues),
            "by_code": by_code,
            "max_track_gap": int(max_track_gap),
        },
        "issues": issues,
    }


def qa_info() -> dict[str, Any]:
    return {
        "pack": EXPORT_QA_ID,
        "box_convention": "normalized_0_1_xywh_top_left",
        "categories": list(CLASSES),
        "checks": [
            "invalid_box",
            "zero_area",
            "out_of_bounds",
            "bad_class",
            "duplicate_box",
            "track_gap",
            "track_fragmented",
        ],
        "defaults": {"max_track_gap": DEFAULT_MAX_TRACK_GAP},
        "endpoints": {
            "GET /api/export/qa": "this document",
            "POST /api/export/qa": (
                "{frames:[{index?,t?,detections:[{class,x,y,w,h,score?,track_id?,source?}]}], "
                "max_track_gap?:int} — validate before/after dataset-pack export; no UI"
            ),
            "GET /api/jobs/{id}/export/qa": "QA a finished job's labels",
        },
        "inspired_by": [
            "https://github.com/Rituparno-Majumdar/annocheck (MIT, annotation QA idea)",
            "COCO / YOLO / MOT public format conventions",
        ],
        "notes": (
            "Backend-only. Does not modify frames or change export formats. "
            "continuity_score is a simple TrackEval-style hint (1.0 = no large track gaps)."
        ),
    }
