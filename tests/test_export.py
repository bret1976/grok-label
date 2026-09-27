"""Dataset pack export tests."""
from __future__ import annotations

import io
import json
import zipfile

from app.export import (
    CATEGORIES,
    mot_track_id,
    to_coco,
    to_mot_pixels,
    to_yolo,
    to_zip_bytes,
)
from app.tracker import assign_tracks


def _walking_person_frames():
    frames = [
        {
            "index": 0,
            "t": 0.0,
            "detections": [
                {"class": "person", "x": 0.20, "y": 0.30, "w": 0.10, "h": 0.40, "score": 0.92},
                {"class": "person", "x": 0.60, "y": 0.28, "w": 0.10, "h": 0.38, "score": 0.88},
            ],
        },
        {
            "index": 1,
            "t": 0.25,
            "detections": [
                {"class": "person", "x": 0.24, "y": 0.31, "w": 0.10, "h": 0.40, "score": 0.91},
                {"class": "person", "x": 0.58, "y": 0.29, "w": 0.10, "h": 0.38, "score": 0.87},
            ],
        },
    ]
    assign_tracks(frames)
    return frames


def test_two_people_persist_ids_across_frames() -> None:
    frames = _walking_person_frames()
    ids0 = {d["track_id"] for d in frames[0]["detections"]}
    ids1 = {d["track_id"] for d in frames[1]["detections"]}
    assert ids0 == ids1 and len(ids0) == 2
    for tid in ids0:
        assert mot_track_id(tid) is not None and mot_track_id(tid) > 0


def test_coco_images_annotations_categories() -> None:
    frames = _walking_person_frames()
    coco = to_coco(frames, width=960, height=540, job_id="testjob")
    assert len(coco["categories"]) == len(CATEGORIES)
    assert coco["categories"][0]["id"] == 1 and coco["categories"][0]["name"] == "person"
    assert len(coco["images"]) == 2
    assert len(coco["annotations"]) == 4
    for ann in coco["annotations"]:
        assert ann["category_id"] == 1
        assert ann["iscrowd"] == 0
        x, y, w, h = ann["bbox"]
        assert w > 0 and h > 0
        assert 0 <= x < 960 and 0 <= y < 540


def test_yolo_lines_normalized() -> None:
    frames = _walking_person_frames()
    files = to_yolo(frames)
    assert "frame_0001.txt" in files and "frame_0002.txt" in files
    for body in files.values():
        for line in body.strip().splitlines():
            parts = line.split()
            assert len(parts) == 5
            cid = int(parts[0])
            assert cid == 0  # person
            vals = [float(p) for p in parts[1:]]
            assert all(0.0 <= v <= 1.0 for v in vals)


def test_mot_integer_ids() -> None:
    frames = _walking_person_frames()
    mot = to_mot_pixels(frames, width=960, height=540)
    lines = [ln for ln in mot.strip().splitlines() if ln]
    assert len(lines) == 4
    ids = set()
    for ln in lines:
        parts = ln.split(",")
        frame_i, tid = int(parts[0]), int(parts[1])
        assert frame_i >= 1
        assert tid > 0
        ids.add(tid)
    assert len(ids) == 2


def test_zip_has_three_formats() -> None:
    frames = _walking_person_frames()
    blob = to_zip_bytes(frames, width=960, height=540, job_id="z1")
    assert blob[:2] == b"PK"
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        names = set(zf.namelist())
        assert "annotations.coco.json" in names
        assert "gt/gt.txt" in names
        assert "labels/frame_0001.txt" in names
        assert "classes.txt" in names
        assert "README.txt" in names
        assert "meta.json" in names
        coco = json.loads(zf.read("annotations.coco.json"))
        assert len(coco["annotations"]) == 4


def test_untracked_low_score_in_coco_not_mot() -> None:
    frames = [
        {
            "index": 0,
            "detections": [
                {"class": "person", "x": 0.2, "y": 0.2, "w": 0.1, "h": 0.3, "score": 0.9},
            ],
        },
        {
            "index": 1,
            "detections": [
                {"class": "person", "x": 0.21, "y": 0.2, "w": 0.1, "h": 0.3, "score": 0.85},
                # low-score unmatched: BYTE will leave track_id None
                {"class": "person", "x": 0.75, "y": 0.4, "w": 0.08, "h": 0.2, "score": 0.15},
            ],
        },
    ]
    assign_tracks(frames)
    untracked = [d for d in frames[1]["detections"] if d.get("track_id") is None]
    assert len(untracked) == 1

    coco = to_coco(frames, width=960, height=540)
    assert len(coco["annotations"]) == 3  # includes untracked

    mot = to_mot_pixels(frames, width=960, height=540)
    mot_lines = [ln for ln in mot.strip().splitlines() if ln]
    assert len(mot_lines) == 2  # only tracked
