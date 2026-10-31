# Final export: 4K / LED (MV_v2, approved 31 Oct 2026)

Everything needed to render the 23 deliverables of the approved cut on any machine (Linux / macOS / Windows).

## Deliverables
| Screen | Render size | Delivery | Format |
|---|---|---|---|
| Main LED | **4608×2592** (16:9 at the LED width) | crop to **4608×2304** in AE: 144 px off top + bottom, layers at 100 % | ProRes 422 HQ, 10-bit, BT.709, 25 fps |
| Side LEDs (L, R) | **832×1456** native | as is; right screen = left mirrored in AE | ProRes 422 HQ |

Every main-screen file has **exactly the length and frame numbering of its approved 1080p preview** (shot list:
`FINAL_CUT_SHOTS.csv`), so in AE: scale the comp to 4608×2592 (File ▸ Scripts ▸ Scale Composition, ×3.6 from a
1280×720 comp), **Replace Footage** with the 4K file, set the layer scale back to its original percentage
(100 % for un-scaled layers). Mirroring / rotation / blending stay in the edit. Frames outside the range the cut uses
(± 10 frame handles) are black, to save render time.

## Jobs (`tools/jobs.py`)
side_rose_transitions, side_rose_loop, dancer_v7, blossom_v4, landscape_transitions, landscape_seamless,
landscape_wobble, curtains_wobble, snow, pigeons, falling_leaf, reflections1, reflections2, hands, couple_dance,
ripples, fireworks, rose_bouquet, leaf3, ballerina_fix, rose3d_transitions, rose3d_loop, fire.
Each entry holds the exact script, inputs, preset and overrides that produced the approved preview.

## Setup on a new machine
1. `git clone` this repo. Python 3.10+ with `numpy opencv-python scipy onnxruntime`, plus `ffmpeg` (with prores_ks) on PATH.
2. Restore the inputs (depth maps, mattes, plates, meshes):
   `bash tools/archive_restore.sh data_archive <WORK>` where `<WORK>` is a work folder (e.g. `work/`).
   Copy all of `pipeline/*.py`, `presets/*.env` and `data/land_gain2.npy` (as `data_land_gain2.npy`) into `<WORK>`.
   Flat depth for the reflections: `python3 -c "import numpy as np,cv2,os;os.makedirs('<WORK>/out/depth_hflat',exist_ok=True);[cv2.imwrite(f'<WORK>/out/depth_hflat/d{i:04d}.png',np.full((720,1280),32767,np.uint16)) for i in range(300)]"`
   The 1080p approved previews (`out/full_new/*.mp4`, used only by `verify`) are not in the repo -- ask for them if you want to verify.
3. Mesh caches: `archive_restore.sh` copies `rosa_mesh.npz` / `redrose_mesh.npz` to `/tmp` (the rose scripts read them from there;
   on Windows edit the `CACHE` path at the top of `rose3d_loop.py` / `rose916.py`).

## Run
```
set WORK=<WORK>  A=<repo>/data_archive/sources  EXPORT=<output folder>
python3 tools/render_queue.py status
python3 tools/render_queue.py run [job ...] --workers <cores> [--budget <minutes>]
python3 tools/render_queue.py verify <job>        # 3 test frames at 1080p vs the approved preview (corr ≥ 0.98 expected)
```
Resumable: finished chunks (`EXPORT/<job>/seg_XXXX.mov`) and jobs (`EXPORT/final/<job>.mov`) are never redone; an
interrupted chunk is simply re-rendered. Peak disk per worker ≈ 40 frames × 30 MB (16-bit PNG) before encoding.
Speed (2-core cloud box): ≈ 30–45 s per 4K frame per worker; ~5,000 main frames + 986 side frames.

## Post steps built in
- Side screens: black point lifted (renderer's 1.8 % red floor → true black), same value for both versions (they stay
  pixel-identical outside the transitions).
- Colour: RGB → Y'CbCr with the BT.709 matrix, tagged 709. (The 1080p H.264 previews were encoded with ffmpeg's
  default BT.601 matrix and no tags, so AE showed them very slightly shifted in hue; the ProRes files are the true colours.)
