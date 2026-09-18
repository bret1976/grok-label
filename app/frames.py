from __future__ import annotations

import json
import subprocess
from pathlib import Path


def extract_frames(video: Path, dest: Path, sample_fps: float) -> list[dict]:
    dest.mkdir(parents=True, exist_ok=True)
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=nw=1:nk=1",
            str(video),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    duration = float(probe.stdout.strip() or 0)
    pattern = dest / "frame_%04d.jpg"
    result = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video),
            "-vf",
            f"fps={sample_fps},scale=960:-1",
            "-q:v",
            "3",
            str(pattern),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr[-800:] or "ffmpeg failed")
    files = sorted(dest.glob("frame_*.jpg"))
    if not files:
        raise RuntimeError("No frames extracted")
    n = len(files)
    frames = []
    for i, path in enumerate(files):
        t = 0.0 if n == 1 else duration * i / max(n - 1, 1)
        frames.append({"index": i, "t": round(t, 3), "path": str(path), "name": path.name})
    meta = {"duration": duration, "sample_fps": sample_fps, "count": n}
    (dest / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return frames
