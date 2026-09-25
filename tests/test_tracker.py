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


def _det(cls, x, y, w=0.08, h=0.12, score=0.9):
    return {"class": cls, "x": x, "y": y, "w": w, "h": h, "score": score}


def test_track_survives_missed_frame_and_gap_is_filled() -> None:
    # A box walks right; Grok misses it in frame 2 (occlusion). Old greedy
    # tracker would spawn a new ID in frame 3.
    frames = [
        {"detections": [_det("cardboard box", 0.10, 0.5)]},
        {"detections": [_det("cardboard box", 0.14, 0.5)]},
        {"detections": []},
        {"detections": [_det("cardboard box", 0.22, 0.5)]},
    ]
    result = assign_tracks(frames)
    ids = {d["track_id"] for f in frames for d in f["detections"]}
    assert ids == {"BOX T0001"}
    assert result["unique_tracks"] == 1
    filled = [d for d in frames[2]["detections"] if d["source"] == "interpolated"]
    assert len(filled) == 1 and abs(filled[0]["x"] - 0.18) < 1e-6
    assert result["tracker"]["refound_after_miss"] == 1


def test_low_score_box_rescues_track_but_cannot_start_one() -> None:
    frames = [
        {"detections": [_det("person", 0.30, 0.3, score=0.9)]},
        {"detections": [_det("person", 0.31, 0.3, score=0.2), _det("person", 0.70, 0.3, score=0.2)]},
    ]
    result = assign_tracks(frames)
    f1 = {round(d["x"], 2): d["track_id"] for d in frames[1]["detections"]}
    assert f1[0.31] == "PERSON T0001"
    assert f1[0.70] is None
    assert result["tracker"]["rescued_low_score"] == 1


def test_motion_prediction_keeps_fast_mover() -> None:
    # Moves 0.07/frame with a 0.08-wide box: IoU with the *previous* box drops
    # to ~0.07, but IoU with the predicted box stays high.
    frames = [{"detections": [_det("cart", 0.05 + 0.07 * i, 0.4)]} for i in range(6)]
    result = assign_tracks(frames)
    assert result["unique_tracks"] == 1


def test_crossing_objects_keep_ids() -> None:
    frames = []
    for i in range(7):
        a = _det("person", 0.10 + 0.05 * i, 0.40)
        b = _det("person", 0.60 - 0.05 * i, 0.44)
        frames.append({"detections": [a, b]})
    assign_tracks(frames)
    first = {d["track_id"] for d in frames[0]["detections"]}
    last = {d["track_id"] for d in frames[-1]["detections"]}
    assert first == last and len(first) == 2
