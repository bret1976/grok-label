"""Dataset pack exporters for Grok Label.

Original helpers that turn labeled frames (normalized 0–1 xywh from top-left)
into COCO instances JSON, YOLO txt labels, MOT Challenge gt, or a zip of all
three. Format specs are public; the multi-format zip idea is inspired by
https://github.com/amanharshx/YOLO-Ndjson-Zip (MIT) — nothing vendored.
"""
from __future__ import annotations

import io
import json
import re
import zipfile
from typing import Any

from app.tracker import CLASSES

CATEGORIES: list[str] = list(CLASSES)

_TRACK_RE = re.compile(r"^(?P<prefix>[A-Z]+)\s+T(?P<num>\d+)\s*$", re.IGNORECASE)
_PREFIX_CLASS = {
    "PERSON": "person",
    "BOX": "cardboard box",
    "TOTE": "plastic tote",
    "CART": "cart",
}


def coco_category_id(cls: str) -> int | None:
    """1-indexed COCO category id, or None if unknown."""
    name = (cls or "").strip().lower()
    try:
        return CATEGORIES.index(name) + 1
    except ValueError:
        return None


def yolo_class_id(cls: str) -> int | None:
    """0-indexed YOLO class id, or None if unknown."""
    name = (cls or "").strip().lower()
    try:
        return CATEGORIES.index(name)
    except ValueError:
        return None


def mot_track_id(track_id: Any) -> int | None:
    """Map string track ids like 'PERSON T0001' to a stable positive int.

    Never returns 0 or -1 for a real track. Unknown / missing -> None.
    """
    if track_id is None:
        return None
    text = str(track_id).strip()
    if not text:
        return None
    m = _TRACK_RE.match(text)
    if m:
        prefix = m.group("prefix").upper()
        num = int(m.group("num"))
        if num <= 0:
            return None
        cls_name = _PREFIX_CLASS.get(prefix)
        if cls_name is None:
            # Still unique: hash prefix into a high band.
            band = (sum(ord(c) for c in prefix) % 90) + 1
            return band * 10000 + num
        return (CATEGORIES.index(cls_name) + 1) * 10000 + num
    # Fallback: stable positive hash for free-form ids.
    h = abs(hash(text)) % 900000 + 1
    return h


def _has_box(det: dict) -> bool:
    try:
        float(det["x"])
        float(det["y"])
        float(det["w"])
        float(det["h"])
        return True
    except (KeyError, TypeError, ValueError):
        return False


def _norm_box(det: dict) -> tuple[float, float, float, float]:
    x = max(0.0, min(1.0, float(det["x"])))
    y = max(0.0, min(1.0, float(det["y"])))
    w = max(1e-6, min(1.0 - x, float(det["w"])))
    h = max(1e-6, min(1.0 - y, float(det["h"])))
    return x, y, w, h


def _frame_number(frame: dict, fallback_i: int) -> int:
    """1-indexed frame id for MOT / YOLO filenames."""
    idx = frame.get("index")
    if isinstance(idx, int):
        return idx + 1
    return fallback_i + 1


def to_coco(
    frames: list[dict],
    *,
    width: int,
    height: int,
    job_id: str | None = None,
) -> dict:
    """COCO instances dict. Includes untracked dets and interpolated boxes."""
    categories = [
        {"id": i + 1, "name": name, "supercategory": "object"}
        for i, name in enumerate(CATEGORIES)
    ]
    images: list[dict] = []
    annotations: list[dict] = []
    ann_id = 1
    for i, frame in enumerate(frames or []):
        fid = _frame_number(frame, i)
        name = frame.get("name") or f"frame_{fid:04d}.jpg"
        images.append(
            {
                "id": fid,
                "file_name": name,
                "width": int(width),
                "height": int(height),
            }
        )
        for det in frame.get("detections") or []:
            if not _has_box(det):
                continue
            cid = coco_category_id(str(det.get("class") or ""))
            if cid is None:
                continue
            x, y, w, h = _norm_box(det)
            bx = x * width
            by = y * height
            bw = w * width
            bh = h * height
            score = float(det.get("score") if det.get("score") is not None else 0.0)
            ann: dict[str, Any] = {
                "id": ann_id,
                "image_id": fid,
                "category_id": cid,
                "bbox": [round(bx, 3), round(by, 3), round(bw, 3), round(bh, 3)],
                "area": round(bw * bh, 3),
                "iscrowd": 0,
                "score": score,
            }
            tid = mot_track_id(det.get("track_id"))
            if tid is not None:
                ann["track_id"] = tid
                ann["track_id_str"] = str(det.get("track_id"))
            if det.get("source"):
                ann["source"] = det["source"]
            annotations.append(ann)
            ann_id += 1
    info: dict[str, Any] = {
        "description": "Grok Label dataset pack",
        "year": 2026,
        "contributor": "Grok Label",
    }
    if job_id:
        info["job_id"] = job_id
    return {
        "info": info,
        "licenses": [{"id": 1, "name": "Unknown", "url": ""}],
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }


def to_yolo(frames: list[dict]) -> dict[str, str]:
    """Map frame_XXXX.txt -> YOLO lines (class cx cy w h normalized)."""
    files: dict[str, str] = {}
    for i, frame in enumerate(frames or []):
        fid = _frame_number(frame, i)
        lines: list[str] = []
        for det in frame.get("detections") or []:
            if not _has_box(det):
                continue
            cid = yolo_class_id(str(det.get("class") or ""))
            if cid is None:
                continue
            x, y, w, h = _norm_box(det)
            cx = x + w / 2
            cy = y + h / 2
            lines.append(
                f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
            )
        files[f"frame_{fid:04d}.txt"] = ("\n".join(lines) + ("\n" if lines else ""))
    return files


def to_mot(frames: list[dict]) -> str:
    """MOT Challenge gt.txt lines: frame,id,bb_left,bb_top,w,h,conf,x,y,z.

    Boxes are converted to pixels with width/height supplied via a caller that
    already scaled them — this function expects detections still normalized and
    encodes bb in *normalized* space only when width/height are 1. Prefer
    ``to_mot_pixels``.
    """
    return to_mot_pixels(frames, width=1.0, height=1.0)


def to_mot_pixels(frames: list[dict], *, width: float, height: float) -> str:
    lines: list[str] = []
    for i, frame in enumerate(frames or []):
        fid = _frame_number(frame, i)
        for det in frame.get("detections") or []:
            if not _has_box(det):
                continue
            tid = mot_track_id(det.get("track_id"))
            if tid is None:
                # MOT requires an id; skip untracked detections.
                continue
            x, y, w, h = _norm_box(det)
            bb_left = x * width
            bb_top = y * height
            bw = w * width
            bh = h * height
            conf = float(det.get("score") if det.get("score") is not None else 1.0)
            lines.append(
                f"{fid},{tid},{bb_left:.3f},{bb_top:.3f},{bw:.3f},{bh:.3f},{conf:.3f},-1,-1,-1"
            )
    return "\n".join(lines) + ("\n" if lines else "")


def build_meta(
    frames: list[dict],
    *,
    width: int,
    height: int,
    job_id: str | None = None,
    format_name: str = "zip",
) -> dict:
    interpolated = 0
    untracked = 0
    total = 0
    for frame in frames or []:
        for det in frame.get("detections") or []:
            if not _has_box(det):
                continue
            total += 1
            if det.get("source") == "interpolated":
                interpolated += 1
            if det.get("track_id") is None:
                untracked += 1
    return {
        "job_id": job_id,
        "format": format_name,
        "box_convention": "normalized_0_1_xywh_top_left",
        "pixel_size": {"width": int(width), "height": int(height)},
        "categories": CATEGORIES,
        "frame_count": len(frames or []),
        "detection_count": total,
        "interpolated_count": interpolated,
        "untracked_count": untracked,
        "notes": (
            "Interpolated gap-fill boxes are included in COCO/MOT/YOLO as real labels "
            "(iscrowd=0). Untracked detections (track_id is None) appear in COCO/YOLO "
            "but are omitted from MOT. Inspired by amanharshx/YOLO-Ndjson-Zip (MIT idea) "
            "+ public COCO / YOLO / MOT specs; original code."
        ),
    }


def to_zip_bytes(
    frames: list[dict],
    *,
    width: int,
    height: int,
    job_id: str | None = None,
) -> bytes:
    coco = to_coco(frames, width=width, height=height, job_id=job_id)
    yolo = to_yolo(frames)
    mot = to_mot_pixels(frames, width=width, height=height)
    meta = build_meta(frames, width=width, height=height, job_id=job_id, format_name="zip")
    readme = (
        "Grok Label dataset pack\n"
        "=======================\n\n"
        "annotations.coco.json  — COCO instances (bbox xywh pixels)\n"
        "labels/*.txt           — YOLO darknet (class cx cy w h normalized)\n"
        "gt/gt.txt              — MOT Challenge (frame,id,bb_left,bb_top,w,h,conf,-1,-1,-1)\n"
        "classes.txt            — class names, YOLO index = line number (0-based)\n"
        "meta.json              — sizes, counts, interpolated/untracked notes\n\n"
        "Boxes in the source job are normalized 0–1 xywh from the top-left.\n"
        "BYTE tracker (byte-v1) is still used for track IDs before export.\n"
        "Multi-format zip idea: amanharshx/YOLO-Ndjson-Zip (MIT). Specs: COCO, YOLO, MOT.\n"
    )
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("annotations.coco.json", json.dumps(coco, indent=2))
        zf.writestr("gt/gt.txt", mot)
        zf.writestr("classes.txt", "\n".join(CATEGORIES) + "\n")
        zf.writestr("README.txt", readme)
        zf.writestr("meta.json", json.dumps(meta, indent=2))
        for name, body in yolo.items():
            zf.writestr(f"labels/{name}", body)
    return buf.getvalue()


def resolve_frame_size(
    frames_dir: Any | None = None,
    *,
    width: int | None = None,
    height: int | None = None,
) -> tuple[int, int]:
    """Prefer explicit size, else first jpeg under frames_dir, else 960×540."""
    if width and height:
        return int(width), int(height)
    w = int(width) if width else 960
    h = int(height) if height else None
    if frames_dir is not None:
        try:
            from pathlib import Path

            from PIL import Image

            root = Path(frames_dir)
            if root.is_dir():
                for path in sorted(root.glob("*.jpg")) + sorted(root.glob("*.jpeg")):
                    with Image.open(path) as im:
                        pw, ph = im.size
                    if not width:
                        w = int(pw)
                    if h is None:
                        h = int(ph)
                    break
        except Exception:
            pass
    if h is None:
        h = 540
    return w, h
