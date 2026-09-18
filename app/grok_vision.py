from __future__ import annotations

import base64
import json
import os
import re
from pathlib import Path
from typing import Any

import httpx

from app.tracker import CLASSES

PROMPT = """You are labeling a warehouse CCTV still for a computer-vision dataset.

Return ONLY JSON:
{"detections":[{"class":"person|cardboard box|plastic tote|cart","x":0.0,"y":0.0,"w":0.0,"h":0.0,"score":0.0}]}

Rules:
- Coordinates are fractions of the FULL image, origin top-left, range 0-1.
- Label every visible person, cardboard box/parcel, blue plastic tote/bin, and wheeled cart.
- Tight boxes. Skip empty floor and furniture.
- If this crop is only part of the warehouse, still use coordinates relative to THIS crop.
- Prefer recall. Overlap is ok; we NMS later.
"""


def resolve_api_key() -> str:
    key = os.environ.get("XAI_API_KEY", "").strip()
    if key:
        return key
    auth = Path.home() / ".grok" / "auth.json"
    if auth.exists():
        try:
            data = json.loads(auth.read_text(encoding="utf-8"))
            for entry in data.values() if isinstance(data, dict) else []:
                token = str((entry or {}).get("key") or "").strip()
                if token:
                    return token
        except (OSError, json.JSONDecodeError):
            pass
    raise RuntimeError("No XAI_API_KEY and no ~/.grok/auth.json session. grok login or set XAI_API_KEY.")


def _client() -> httpx.AsyncClient:
    key = resolve_api_key()
    return httpx.AsyncClient(
        base_url="https://api.x.ai/v1",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        timeout=180.0,
    )


def _encode(path: Path) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def _parse(text: str) -> list[dict[str, Any]]:
    raw = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", raw, re.S)
    if fence:
        raw = fence.group(1)
    start, end = raw.find("{"), raw.rfind("}")
    if start >= 0 and end > start:
        raw = raw[start : end + 1]
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return []
    out = []
    for item in payload.get("detections") or []:
        cls = str(item.get("class") or "").strip().lower()
        if cls in {"box", "parcel", "cardboard"}:
            cls = "cardboard box"
        if cls in {"tote", "bin", "plastic bin"}:
            cls = "plastic tote"
        if cls in {"worker", "employee", "human"}:
            cls = "person"
        if cls in {"trolley", "rolling cart"}:
            cls = "cart"
        if cls not in CLASSES:
            continue
        try:
            x, y, w, h = float(item["x"]), float(item["y"]), float(item["w"]), float(item["h"])
        except (KeyError, TypeError, ValueError):
            continue
        x, y = max(0.0, min(x, 1.0)), max(0.0, min(y, 1.0))
        w, h = max(0.01, min(w, 1.0 - x)), max(0.01, min(h, 1.0 - y))
        out.append(
            {
                "class": cls,
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "score": float(item.get("score") or 0.6),
            }
        )
    return out


def _shift(dets: list[dict], ox: float, oy: float, sx: float, sy: float) -> list[dict]:
    shifted = []
    for det in dets:
        shifted.append(
            {
                **det,
                "x": ox + det["x"] * sx,
                "y": oy + det["y"] * sy,
                "w": det["w"] * sx,
                "h": det["h"] * sy,
            }
        )
    return shifted


async def label_image(path: Path, *, model: str, tile_note: str = "full frame") -> list[dict[str, Any]]:
    body = {
        "model": model,
        "temperature": 0.0,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": _encode(path), "detail": "high"}},
                    {"type": "text", "text": PROMPT + f"\nThis crop: {tile_note}."},
                ],
            }
        ],
    }
    async with _client() as client:
        response = await client.post("/chat/completions", json=body)
    if response.status_code >= 400:
        raise RuntimeError(f"Grok vision error {response.status_code}: {response.text[:400]}")
    text = (((response.json().get("choices") or [{}])[0].get("message") or {}).get("content")) or ""
    return _parse(text)


async def label_frame(path: Path, *, model: str, grid: int) -> list[dict[str, Any]]:
    from PIL import Image

    dets = await label_image(path, model=model, tile_note="full warehouse CCTV")
    if grid <= 1:
        return dets
    img = Image.open(path)
    w, h = img.size
    cell_w, cell_h = w // grid, h // grid
    tmp = path.parent / "_tiles"
    tmp.mkdir(exist_ok=True)
    for gy in range(grid):
        for gx in range(grid):
            box = (gx * cell_w, gy * cell_h, w if gx == grid - 1 else (gx + 1) * cell_w, h if gy == grid - 1 else (gy + 1) * cell_h)
            tile = tmp / f"{path.stem}_{gy}_{gx}.jpg"
            img.crop(box).save(tile, quality=85)
            ox, oy = box[0] / w, box[1] / h
            sx, sy = (box[2] - box[0]) / w, (box[3] - box[1]) / h
            tile_dets = await label_image(tile, model=model, tile_note=f"tile row {gy} col {gx}")
            dets.extend(_shift(tile_dets, ox, oy, sx, sy))
    return dets
