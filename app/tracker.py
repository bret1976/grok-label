from __future__ import annotations

from typing import Any

CLASSES = ("person", "cardboard box", "plastic tote", "cart")
CLASS_PREFIX = {
    "person": "PERSON",
    "cardboard box": "BOX",
    "plastic tote": "TOTE",
    "cart": "CART",
}


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


def assign_tracks(frames: list[dict[str, Any]], iou_thresh: float = 0.28) -> dict[str, Any]:
    counters = {name: 0 for name in CLASSES}
    live: list[dict] = []
    for frame in frames:
        used: set[int] = set()
        next_live: list[dict] = []
        dets = nms(frame.get("detections") or [])
        for det in sorted(dets, key=lambda d: float(d.get("score") or 0), reverse=True):
            best_i, best = -1, iou_thresh
            for i, track in enumerate(live):
                if i in used or track["class"] != det["class"]:
                    continue
                score = iou(track, det)
                if score > best:
                    best, best_i = score, i
            if best_i >= 0:
                track = live[best_i]
                used.add(best_i)
                det["track_id"] = track["track_id"]
                next_live.append({**det})
            else:
                counters[det["class"]] = counters.get(det["class"], 0) + 1
                prefix = CLASS_PREFIX.get(det["class"], "OBJ")
                det["track_id"] = f"{prefix} T{counters[det['class']]:04d}"
                next_live.append({**det})
            det["source"] = "detected"
        frame["detections"] = dets
        live = next_live

    unique: dict[str, dict] = {}
    for frame in frames:
        for det in frame.get("detections") or []:
            tid = det.get("track_id")
            if not tid:
                continue
            unique.setdefault(tid, {"track_id": tid, "class": det["class"], "frames": 0})
            unique[tid]["frames"] += 1
    return {
        "tracks": list(unique.values()),
        "class_counts": {k: v for k, v in counters.items()},
        "unique_tracks": len(unique),
    }


def stats(frames: list[dict], tracks: list[dict]) -> dict[str, int]:
    by_class = {name: 0 for name in CLASSES}
    for track in tracks:
        cls = track.get("class")
        if cls in by_class:
            by_class[cls] += 1
    total_dets = sum(len(f.get("detections") or []) for f in frames)
    return {
        "all": sum(by_class.values()),
        "carts": by_class["cart"],
        "parcels": by_class["cardboard box"] + by_class["plastic tote"],
        "employees": by_class["person"],
        "annotations": total_dets,
        "classes": by_class,
    }
