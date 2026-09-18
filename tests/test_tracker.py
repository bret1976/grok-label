from app.tracker import assign_tracks, iou, nms


def test_nms_drops_overlap() -> None:
    dets = [
        {"class": "cart", "x": 0.1, "y": 0.1, "w": 0.2, "h": 0.2, "score": 0.9},
        {"class": "cart", "x": 0.12, "y": 0.12, "w": 0.2, "h": 0.2, "score": 0.4},
        {"class": "person", "x": 0.12, "y": 0.12, "w": 0.2, "h": 0.2, "score": 0.8},
    ]
    kept = nms(dets, 0.5)
    assert len(kept) == 2


def test_tracks_persist_across_frames() -> None:
    frames = [
        {
            "detections": [
                {"class": "person", "x": 0.2, "y": 0.2, "w": 0.1, "h": 0.3, "score": 0.9}
            ]
        },
        {
            "detections": [
                {"class": "person", "x": 0.22, "y": 0.21, "w": 0.1, "h": 0.3, "score": 0.8}
            ]
        },
    ]
    result = assign_tracks(frames)
    assert frames[0]["detections"][0]["track_id"] == frames[1]["detections"][0]["track_id"]
    assert result["unique_tracks"] == 1
    assert iou(frames[0]["detections"][0], frames[1]["detections"][0]) > 0.5
