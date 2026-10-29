# Shot runbook

`A` = folder holding the uploaded source clips (in the original session: `/mnt/user-data/attachments/<chat-id>/`;
file names below keep the upload's own name, prefixed by an upload id — match on the part after the first `-`).
All commands run from `$WORK` (see README). Add `OUTS=3` for 4K. Any script that writes a video through
`common.Writer` writes a **16-bit PNG sequence** instead when `PNGDIR=<dir>` (and optional `PNGNAME`, `PNGOFF`)
is set; scripts with `STILL=<prefix>` / an output dir already write PNGs.

**Status legend** — ✅ exact command recorded · 🔶 script + preset + source are exact, frame range / args
reconstructed: render 3 test frames and compare with the approved 1080p file before the 4K run.

| # | Shot (approved file) | Status |
|---|---|---|
| 1 | Couple dance — `dance_full_1080p.mp4` | ✅ |
| 2 | Gulls — `gulls_full_1080p.mp4` | ✅ |
| 3 | Rose fly-through — `rose_fly2_1080p.mp4` | ✅ |
| 4 | Rose bouquet — `rose_LOCKED_1080p.mp4` | 🔶 |
| 5 | Landscape loop — `landscape_loop15_seamless.mp4` / `_transitions.mp4` | 🔶 |
| 6 | Dancer — `dancer_full_720p_v6.mp4` | 🔶 |
| 7 | Hands — `hands_full_1080p_v3.mp4` | 🔶 |
| 8 | Blossom — `blossom_full_1080p.mp4` | 🔶 |
| 9 | Branch — `branch_full_1080p_v3.mp4` | 🔶 |
| 10 | Leaf (falling) — `leaf_full_720p.mp4`, Leaf 3 — `leaf3_full_1080p_v3.mp4` | 🔶 |
| 11 | Water / dappled light — `water_full_720p.mp4` | 🔶 |
| 12 | Fireworks — `fireworks_full_720p.mp4` | 🔶 |
| 13 | Pigeons — `pigeons_full_720p_v2.mp4` | 🔶 |
| 14 | Snow — `snow_full_1080p.mp4` | 🔶 |
| 15 | Silk / vertical waves (rough cut 1:46–2:12) | 🔶 look scripts (`E_silk_loop*`, `pipeline/silk*.py`) |
| 16 | Stars / Milky Way — `stars_full_1080p.mp4` (dropped, not approved) | 🔶 |

**Keep the approved 1080p files** — they are the reference for every 🔶 verification.

---
## 1 · Couple dance ✅
- Source: `e2159d18-coupleshot_2.mp4` (1920×1080, 25 fps) — frames 0–214 (8 s 15 f).
- Script: `motion_sil3.py` → `sil_bg3.py` (one dot sheet: curved sheet of circles, dancers carved out, edge dots lit
  by the backlight, burst-in from the centre, peel-out along the background flow). No depth maps needed.
- Preset: `presets/dance_full.env`
```bash
env $(cat dance_full.env) OUTS=3 python3 motion_sil3.py $A/e2159d18-coupleshot_2.mp4 0 215 out/dance4k 0 108 &
env $(cat dance_full.env) OUTS=3 python3 motion_sil3.py $A/e2159d18-coupleshot_2.mp4 0 215 out/dance4k 108 215
```
(two halves in parallel; the second warms up 40 frames before 108 so the light smoothing matches.)

## 2 · Gulls ✅
- Source: `dea9f544-3327030-hd_1920_1080_24fps.mp4` (24 fps, played frame-for-frame at 25 → 252 frames, 10.08 s).
- Script: `gulls_full.py` (keyed birds, per-bird dot grids via `depth_sheet_objects.py`, tracking crop 720→1060 px,
  flight-path forming / dispersal, roof + vignette corners removed). No depth maps. **Run in ONE process** (tracking).
- Preset: `presets/gulls_sf.env` minus its `CROP=` entry.
```bash
E=$(sed 's/CROP=420,80,1000//' gulls_sf.env)
env $E OUTS=3 python3 gulls_full.py $A/dea9f544-3327030-hd_1920_1080_24fps.mp4 out/gulls4k 0 252
```
- Post step (applied to the approved version): fade frames 228–251 to black and crush the faint red remnant:
```python
for i in range(228,252):  # a = frame as float 0..1
    x=(251-i)/23; k=x*x*(3-2*x); lift=0.03*(1-k); a=np.clip((a-lift)/(1-lift),0,1)*k
```

## 3 · Rose fly-through ✅
- Source: `02f87d17-7675962-hd_1920_1080_25fps.mp4` (276 frames) — uses frames 0–250 (10.04 s).
- Depth: `out/depth_rose` (276 frames) — rebuild:
```bash
ffmpeg -i $A/02f87d17-7675962-hd_1920_1080_25fps.mp4 -vf "scale=1280:720:flags=area" -an -c:v libx264 -crf 12 depthjobs/rose_full.mp4
python3 depth_full2.py depthjobs/rose_full.mp4 depthjobs/rose_fraw 100000 0 640 360
python3 depth_stab2.py depthjobs/rose_full.mp4 depthjobs/rose_fraw out/depth_rose 4 0.04 0.25 1920 1080
```
- Script: `depth_sheet_clip.py` (CAMPATH=rose2: continuous orbit + push, focus pull, DEEP extra dot level,
  staggered burst). Preset: `presets/rose_fly2_preset.env`
```bash
for r in 0,140 140,251; do env $(cat rose_fly2_preset.env) OUTS=3 RANGE=$r STILL=out/rosefly4k/rose_fly2 \
  python3 depth_sheet_clip.py out/depth_rose 0 251 x.mp4 $A/02f87d17-7675962-hd_1920_1080_25fps.mp4 0 & done
```

## 4 · Rose bouquet (LOCKED) 🔶
- Same source + depth as #3. Script `depth_sheet_clip.py`, preset `presets/rose_preset_LOCKED.env` (CLIPLEN 11.04 → 276 frames).
```bash
env $(cat rose_preset_LOCKED.env) OUTS=3 STILL=out/rose4k/rose python3 depth_sheet_clip.py out/depth_rose 0 276 x.mp4 $A/02f87d17-7675962-hd_1920_1080_25fps.mp4 0
```

## 5 · Landscape loop 🔶
- Procedural, no footage. Script `pipeline/land/landLoop.py outdir f0 f1` (15 s = 375 frames, seamless).
- Preset `presets/land_loop_final.env` (seamless; uses `data/land_gain2.npy` via GAINFILE). Transitions version:
  add `LOOPMODE=trans`.
```bash
cp pipeline/land/landLoop.py . ; env $(cat land_loop_final.env) OUTS=3 python3 landLoop.py out/land4k 0 375
```
(split into chunks across the 2 cores; chunks are independent.)

## 6 · Dancer 🔶
- Source: the 2048×1080 dancer clip (`8ba4048d-8462446-hd_2048_1080_25fps.mp4`, 600 frames), seconds 3–18
  (frames 75–449), centre-cropped to 16:9 → `out/dancer_3-18s_1920.mp4`; depth `out/depth_dancer_v3` (375 frames).
- Script `depth_sheet_clip.py`, preset `presets/dancer_preset_v6.env` (+ transition settings as in the approved v6).

## 7 · Hands 🔶
- Depth `out/depth_hands` (209 frames). Script `depth_sheet_clip.py`, preset `presets/hands_preset.env`
  (CLIPLEN 8.36, TRANS_MODE=hands, EDGEMATTE).

## 8 · Blossom 🔶
- Footage conformed with `conform_dedupe.py` (held frames removed) → `out/blossom_footage_25fps_1920.mp4`;
  depth `out/depth_blossom` (337 frames). Script `clip_new.py … depthsubj`, preset `presets/blossom_preset_trans.env`
  (BRANCH_DETAIL cylinder shading, particle forming / dispersing).

## 9 · Branch 🔶
- Source `cc8bdf38-golden_branch_cropped.mp4` (371 frames) → `branch_stab.py` (camera stabilisation + light
  smoothing) → `out/branch_stab2_25fps_1920.mp4` + `out/depth_branch_stab2`. Script `clip_new.py … depthsubj`,
  preset `presets/branch_preset_v3.env` (CLIPLEN 14.84).

## 10 · Leaves 🔶
- Leaf: `0b8ccd2a-falling_leaf_1.mp4` (97 frames), `run_new3_full.sh` recipe for depth; `clip_new.py … leaf`,
  preset `presets/leaf_preset.env` (CLIPLEN 3.88, TRAIL sparkles).
- Leaf 3: `692c292a-falling_leaf_3.mp4` (24 fps, conformed) → `out/depth_leaf3` (149); `clip_new.py … scene`,
  preset `presets/leaf3_preset_trans.env` (CLIPLEN 5.96).

## 11 · Water / dappled light 🔶
- `21c78dbe-8348420-hd_2048_1080_25fps.mp4` (crop 1920:1080:64:0) — depth recipe in `run_new3_full.sh`;
  `clip_new.py … water`, preset `presets/water_preset.env` (CLIPLEN 15.48, soft transition).

## 12 · Fireworks 🔶
- `883e5634-fireworks.mp4` (370 frames) — depth recipe in `run_new3_full.sh`; `clip_new.py … fireworks`,
  preset `presets/fireworks_preset.env` (CLIPLEN 14.8).

## 13 · Pigeons 🔶
- `b264125e-12772675_1920_1080_25fps.mp4` (288 frames) → `out/depth_pigeons_v3`; `depth_sheet_objects.py`
  (per-bird grids, flow transition), preset `presets/pigeon_preset.env`, `TRANS=1`.

## 14 · Snow 🔶
- `60b9fb00-snow.mp4` (369 frames); `clip_new.py <src> none 0 369 snow out.mp4`, preset `presets/snow_preset.env`
  (no depth: every flake is a particle).

## 15 · Silk / vertical waves 🔶
- Procedural loop from look development (`pipeline/silk*.py`, output was `E_silk_loop_v2.mp4`). Identify the exact
  variant by comparing with the cut before re-rendering.

## 16 · Stars 🔶 (dropped)
- `d9b9682d-7437640-uhd_4096_1974_30fps.mp4`; `stars_clip.py src out.mp4`, preset `presets/stars_preset_nebula.env`.

---
### Still to build (as of 29 Oct 2026)
- Kiss silhouette (`9ff6734f-couple_fix_2.mp4`) in the dance look (`sil_bg3.py`).
- 3D rose loop from `rosa.obj` (+ textures), 15 s, 4K — stretch goal.
- Song section timecodes → shot list for the locked edit.

## 17 · 3D rose (style frame stage) 🔶
- Model: `rosa.obj` + `rosa.obj.mtl` + albedo texture `rosa.jpg` (the pink/green petal atlas; uploaded as an image).
- `rose3d.py rosa.obj rosa.jpg out.png [t]` — point-splat z-buffer of the mesh → picture + depth → `depth_sheet.render`.
  Z is the model's up axis (`R_UP=0,0,1`); turntable angle `R_YAW` (deg). Preset `presets/rose3d_sf.env` (yaw 190°).
- Loop: `rose3d_loop.py rosa.obj rosa.jpg outdir f0 f1`, preset `presets/rose3d_loop.env` (T = 24 beats = 9.86 s,
  one turn + 2 particle waves per loop, seamless). Full loop = frames 0–245 (9.84 s, 24 beats; split 0–123 / 123–246 across 2 processes). 2 s test = frames 20–69.
- v2 (2× rotation + orbiting comet streams): preset `presets/rose3d_loop2.env`; with transitions in/out:
  `presets/rose3d_trans.env` (frames 0–245, first/last frame black). Run each in ONE process (frames 0 246).

## 18 · Ballerina 2 (test stage) 🔶
- Source `82403cef-8462335-hd_2048_1080_25fps.mp4` (802 fr) → crop 1920:1080:64:0 → `out/bal2_1920.mp4`; depth as usual
  (`depth_full2.py` + `depth_stab2.py` → `out/depth_bal2`). `bal2_prep.py` keys out the white wall (depth vs fitted wall
  + colour edge) and relights her like ballerina 1 (black clothes → glowing garment, side key) → relit video on black;
  then `depth_sheet_clip.py out/depth_bal2 <a> <n> out.mp4 relit.mp4 <-offset>` with `presets/bal2_preset.env`
  (= dancer v6 + ORBIT_W 0.22). Test: frames 150–199 (B_SKIN=0.9).

## 19 · Fire (test stage) 🔶
- Source `9d8441a2-fire.mp4` (377 fr). `fire_clip.py src outdir f0 f1`, preset `presets/fire_preset.env`
  (no depth model; embers from the flame top; TRANS=1 → in from below / out upwards via depth_sheet `odir`). Test: 100–149.
- Fire full: `env $(cat fire_full.env) python3 fire_clip.py <src> out/fire_full 0 377` (one process).
- Ballerina 2 full (source 4–19 s): `B_SKIN=0.9 python3 bal2_prep.py out/bal2_1920.mp4 out/depth_bal2 out/bal2_relit_100.mp4 100 475`,
  then `env $(cat bal2_full.env) STILL=out/bal2_full/bal2 python3 depth_sheet_clip.py out/depth_bal2 100 375 x.mp4 out/bal2_relit_100.mp4 -100`.
