"""Export QA (export-qa-v1) tests — backend only."""
from __future__ import annotations

from app.export_qa import EXPORT_QA_ID, qa_info, validate_frames
from app.tracker import assign_tracks


def _clean_pair():
    frames = [
        {
            "index": 0,
            "t": 0.0,
            "detections": [
                {"class": "person", "x": 0.20, "y": 0.30, "w": 0.10, "h": 0.40, "score": 0.9},
                {"class": "cart", "x": 0.55, "y": 0.40, "w": 0.20, "h": 0.25, "score": 0.85},
            ],
        },
        {
            "index": 1,
            "t": 0.25,
            "detections": [
                {"class": "person", "x": 0.22, "y": 0.30, "w": 0.10, "h": 0.40, "score": 0.88},
                {"class": "cart", "x": 0.56, "y": 0.41, "w": 0.20, "h": 0.25, "score": 0.84},
            ],
        },
    ]
    assign_tracks(frames)
    return frames


def test_clean_frames_pass() -> None:
    report = validate_frames(_clean_pair())
    assert report["pack"] == EXPORT_QA_ID
    assert report["ok"] is True
    assert report["summary"]["errors"] == 0
    assert report["summary"]["continuity_score"] == 1.0
    assert report["summary"]["detections"] == 4


def test_zero_area_and_oob_and_bad_class() -> None:
    frames = [
        {
            "index": 0,
            "detections": [
                {"class": "person", "x": 0.1, "y": 0.1, "w": 0.0, "h": 0.2, "score": 0.9},
                {"class": "alien", "x": 0.2, "y": 0.2, "w": 0.1, "h": 0.1, "score": 0.9},
                {"class": "cart", "x": 0.95, "y": 0.95, "w": 0.2, "h": 0.2, "score": 0.9},
            ],
        }
    ]
    report = validate_frames(frames)
    assert report["ok"] is False
    codes = {i["code"] for i in report["issues"]}
    assert "zero_area" in codes
    assert "bad_class" in codes
    assert "out_of_bounds" in codes


def test_duplicate_box_warning() -> None:
    frames = [
        {
            "index": 0,
            "detections": [
                {"class": "person", "x": 0.2, "y": 0.2, "w": 0.1, "h": 0.3, "score": 0.9},
                {"class": "person", "x": 0.201, "y": 0.201, "w": 0.1, "h": 0.3, "score": 0.8},
            ],
        }
    ]
    report = validate_frames(frames)
    assert any(i["code"] == "duplicate_box" for i in report["issues"])
    assert report["summary"]["warnings"] >= 1


def test_track_gap_warning() -> None:
    # Same track_id at frame 0 and frame 12 with nothing in between → gap 11 > 8.
    frames = [
        {
            "index": 0,
            "detections": [
                {
                    "class": "person",
                    "x": 0.2,
                    "y": 0.2,
                    "w": 0.1,
                    "h": 0.3,
                    "score": 0.9,
                    "track_id": "PERSON T0001",
                    "source": "detected",
                }
            ],
        },
        *[{"index": i, "detections": []} for i in range(1, 12)],
        {
            "index": 12,
            "detections": [
                {
                    "class": "person",
                    "x": 0.25,
                    "y": 0.2,
                    "w": 0.1,
                    "h": 0.3,
                    "score": 0.9,
                    "track_id": "PERSON T0001",
                    "source": "detected",
                }
            ],
        },
    ]
    report = validate_frames(frames, max_track_gap=8)
    assert report["ok"] is True  # gaps are warnings
    assert any(i["code"] == "track_gap" for i in report["issues"])
    assert report["summary"]["track_gap_events"] == 1
    assert report["summary"]["continuity_score"] < 1.0


def test_qa_info_pack() -> None:
    info = qa_info()
    assert info["pack"] == EXPORT_QA_ID
    assert "invalid_box" in info["checks"]
    assert "POST /api/export/qa" in info["endpoints"]
