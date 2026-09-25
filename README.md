# Grok Label

Warehouse CCTV auto-labeling — the Higgsfield / GPT-6 Astra “warehouse, frame by frame” lab, rebuilt on the **Grok vision API**.

Reference: [higgsfield_ai/status/2097002211607982463](https://x.com/higgsfield_ai/status/2097002211607982463)

## What the demo actually is

The X video is silent (no audio track). It is a **screen recording of ChatGPT (GPT-6 Astra Ultra)** driving a Higgsfield site:

`warehouse-observation-lab.higgsfield-i-1075.chatgpt.site`

Pipeline on screen: **Generate → Annotate → Inspect**.

| Claim in the tweet | Measured from the clip |
| --- | --- |
| 20.6s warehouse CCTV | 2288×1608, 24 fps, **no audio** |
| 81 annotated frames | ~4 fps sampling of a 20s clip |
| 13,038 annotations | ~160 boxes/frame (peak ~181) |
| Classes | person, cardboard box, plastic tote, cart |
| Track IDs | `PERSON T0013`, `BOX T0394`, `TOTE T0285`, `CART T0106` |
| UI | overlay / labels / trails / interpolation, Original vs Annotated |

Comments worth building for:

- Occlusion when parcels overlap (`@unfairstack`)
- Tracking from a few analyzed frames + interpolation (`@cdev_ai`)
- Density / efficiency vs an army of labelers

## This repo

Grok does the labeling. No Higgsfield.

1. Upload CCTV (or use the downloaded sample clip).
2. Sample frames (`SAMPLE_FPS`, default 4).
3. **Grok vision** (`XAI_API_KEY`, default `grok-4.5`) returns boxes per full frame **and** a tile grid so small parcels are not skipped.
4. **BYTE tracker** (ByteTrack-style, see below) assigns stable IDs; the player interpolates between sampled frames.
5. Inspect UI matches the demo: class cards, overlay, labels, trails, interpolation.

```bash
cd grok-label
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # put XAI_API_KEY here
uvicorn app.main:app --reload --port 8787
```

Open http://127.0.0.1:8787 — **Annotate sample CCTV with Grok**.

Env:

| Variable | Meaning |
| --- | --- |
| `XAI_API_KEY` | https://console.x.ai |
| `GROK_MODEL` | default `grok-4.5` |
| `SAMPLE_FPS` | frames sent to Grok (default 4 → ~81 frames on a 20s clip) |
| `TILE_GRID` | 1 = full frame only; 2 = 2×2 extra tiles |
| `MAX_CONCURRENCY` | parallel Grok calls |

Screenshots from the original post live in `research/`.

## Tracking (BYTE, ByteTrack-style)

`app/tracker.py` is an original pure-Python take on the BYTE association from
[FoundationVision/ByteTrack](https://github.com/FoundationVision/ByteTrack) (MIT):

- high-confidence boxes match first against active **and recently lost** tracks, using a
  constant-velocity predicted box (plus a centre-distance fallback for fast movers at 4 fps);
- low-confidence boxes (usually occluded / overlapping parcels) can only *continue* a track, never start one;
- lost tracks survive `track_buffer` sampled frames (default 8 ≈ 2 s), so one missed frame no longer spawns a new ID;
- short gaps are filled with interpolated boxes (`source: "interpolated"`).

On a synthetic 160-object / 81-frame warehouse clip with 15% of boxes missed per frame,
the old greedy IoU tracker produced 1,336 IDs; the BYTE tracker produces 179.

API:

| Route | Use |
| --- | --- |
| `GET /api/tracker` | algorithm + default params |
| `POST /api/track` | track your own detections `{frames:[{t, detections:[{class,x,y,w,h,score}]}], params:{}}` — no Grok call |
| `POST /api/jobs/{id}/retrack` | re-run tracking on a finished job with new params — no Grok call |
