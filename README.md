# Dukla — "Den, kdy se vrátí láska" music video: dot-sheet pipeline

Everything needed to re-render every shot of the video (1080p previews and the final **4K** export) in the
locked "D" look: gold / white-silver / red-burgundy halftone dot sheet, glow, sparse glints, 146 BPM.

The repo holds **code + per-shot settings + this runbook**. It does NOT hold source footage, depth maps or renders:
- **source footage** — the user keeps the original clips; re-upload the ones a shot needs (see `SHOTS.md`).
- **depth maps** — regenerated from the footage with `pipeline/depth_full2.py` + `depth_stab2.py` (deterministic).
- **renders** — delivered to the user shot by shot.

## Layout
| path | what |
|---|---|
| `pipeline/` | all render scripts (Python). Run them from a working dir laid out like `WORK` below |
| `pipeline/land/` | landscape / wave loop (`landLoop.py`) and its helpers |
| `pipeline/look/` | look-development scripts (style frames, debugging) |
| `presets/*.env` | one file per shot: every look setting as `KEY=value` env vars |
| `data/` | small data: landscape exposure curve (`land_gain2.npy`), beat grid (`beats.npy`) |
| `archive_backups/` | earlier versions of scripts (kept for reference only) |
| `SHOTS.md` | **per-shot runbook**: source clip, script, preset, frame range, exact commands |

## Setup in a fresh session (≈10 min)
```bash
git clone https://github.com/ycyprid/dukla-music-video-1 repo
WORK=/home/claude/mv; mkdir -p $WORK/models $WORK/out $WORK/depthjobs
cp repo/pipeline/*.py repo/pipeline/*.sh $WORK/ ; cp repo/presets/*.env $WORK/ ; cp -r repo/data $WORK/
pip install --break-system-packages numpy opencv-contrib-python scipy onnxruntime   # tested: numpy 2.5, opencv 5.0, scipy 1.18, onnxruntime 1.29, py 3.13
# ffmpeg must be on PATH (tested 6.1)
```
**Depth model** (99 MB, not in the repo): Depth Anything V2 *Small*, ONNX with dynamic input size, saved as
`$WORK/models/depth_anything_v2_vits_dynamic.onnx`. Public release: github.com/fabio-sim/Depth-Anything-ONNX
(V2 small, dynamic). sha256 of the file we used:
`46c4e8eeda3a27f34701831b6a2ec7753d7b38779b215acb5633424703deed8f` (99,092,268 bytes) — check it matches.

Scripts assume they run from `$WORK` (relative paths `models/`, `out/`, `depthjobs/`) and write scratch files to `/tmp`.

## Core ideas (so changes stay consistent)
- `depth_sheet.py` is the renderer. Design space is 1280×720 (`common.W,H`); **`OUTS` sets the output scale**:
  `OUTS=1.5` → 1920×1080, **`OUTS=3` → 3840×2160 (4K)**. Dots are drawn at the output resolution, so 4K is real
  detail, not an upscale. Nothing else changes between 1080p and 4K — same presets, add/override `OUTS=3`.
- Look settings are env vars (presets). Run a shot with `env $(cat presets/X.env) OUTS=3 python3 script.py …`.
- Time is in seconds at **25 fps**; beat grid 146 BPM (`BT = 60/146`).
- Depth for footage shots: `depth_full2.py` (raw per-frame) → `depth_stab2.py` (flow-guided temporal smoothing) →
  `out/depth_<shot>/d0000.png …` (16-bit).

## 4K export procedure (per shot)
1. Re-upload that shot's source clip(s). Rebuild its depth if the shot uses depth (see SHOTS.md).
2. Run the shot's command with `OUTS=3` and `STILL=` / PNG output → 16-bit PNG sequence (≈25 MB/frame).
3. **Verify** a few frames against the approved 1080p version (same frame numbers, downscaled diff).
4. Deliver (link the chat to the user's computer and write into their folder, or package and send), then delete
   it here before the next shot — the workspace has ≈40 GB in total.
Render speed at 4K ≈ 10–15 s per frame per process; with 2 processes ≈ 8–12 frames a minute, i.e. about 2–3 min of rendering per second of video.
