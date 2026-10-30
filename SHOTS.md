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

## Edit v1_1 fixes (30 Oct)
- **Blossom v2**: `blossom_matte.py <footage> out/depth_blossom 337 out/blossom_matte` (flow-aligned temporal median of
  the matte + in-focus gate + hole restore), then `clip_new.py ... depthsubj` with `MATTE_DIR=out/blossom_matte`.
- **Landscape wobble (cut 0:33)**: `landLoop_wob.py` (= landLoop + WOB_T0/WOB_A/WOB_S ramp, ONLY=, NO_IN, OUT_MIN),
  `env $(cat land_wob.env) LOOPMODE=trans WOB_T0=6.62 python3 landLoop_wob.py out/land_wob 150 375` → replaces the
  2nd pass of `landscape_loop15_transitions` from its frame 150 (cut 32.48 s).
- **Curtains wobble (cut 1:57)**: `... LOOPMODE=trans WOB_T0=12.2 NO_IN=1 OUT_MIN=15 ... out/curt_wob 305 750`
  (continuous: loop frame 305 = cut 117.0 s → end of the out-transition, cut 134.8 s).
- **3D rose**: both presets now pin `RIMS` (rim scale); the transitions version had been calibrated on its black
  first frame → ~6 % brighter dots.
- **Ripples 7.5 s**: `clip_new.py out/water_footage_25fps_1920.mp4 out/depth_water 0 188 water` with water preset +
  `CLIPLEN=7.52 SOFT_K=0.55`.
- **Blossom v3 (final)**: the old footage had been made with ffmpeg minterpolate (336 unique frames at 24.8 fps →
  25 fps), which warped / smeared petals and caused stutter. Now: source `d9b9682d-7437640-uhd_4096_1974_30fps.mp4`,
  centre-crop 16:9, held frames dropped, the 336 unique frames played 1:1 at 25 fps (+1 hold at the end),
  irregular first 7 frames rebuilt from frame 7 (flow extrapolation) → `out/blossom_footage_v2.mp4` (+1280 copy for depth).
  New depth `out/depth_blossom2`; matte `MT_OPEN=3 MT_R=3 blossom_matte.py ... out/blossom_matte3`;
  render `clip_new.py out/blossom_footage_v2.mp4 out/depth_blossom2 0 337 depthsubj` with `MATTE_DIR=out/blossom_matte3`.
- **Blossom v4 (current)**: pops in v3 came from hard per-frame matte decisions (hole fill, focus gate, thresholds).
  `blossom_matte.py` now ends with a bidirectional flow-guided recursive average (MT_A=0.82, fwd/bwd consistency trust)
  + soft threshold → pops become fades. `MT_OPEN=3 python3 blossom_matte.py out/blossom_footage_v2.mp4 out/depth_blossom2 337 out/blossom_matte4`
  (matte pop area 24,066 → 135 px). Depth smoothed the same way: `python3 depth_tfilter.py ... out/depth_blossom2s` (A=0.75).
  Render: `env $(cat presets/blossom_preset_trans.env) MATTE_DIR=out/blossom_matte4 PNGDIR=out/blossom_v4 PNGNAME=blossom
  python3 clip_new.py out/blossom_footage_v2.mp4 out/depth_blossom2s 0 337 depthsubj x.mp4`
- **Fire v2 (4 s 20 f)**: `env $(cat presets/fire_v2.env) python3 fire_clip.py <src> out/fire_v2 0 120` — GAIN 1.7,
  `HOTW=0.72,0.85` (new depth_sheet option: brightest dots blend to white-silver), `FIRE_TL=4.8` moves the out-transition
  to start at 2.6 s.

## 20 · Hands silhouette + Ballerina 3 (test stage) 🔶
- Tests = source 4–6 s (`ffmpeg -ss 4 -frames:v 50`; ballerina 3 = `23c9e9be-8462217` centre-cropped 2048→1920).
  Depth: `depth_full2.py … 1280 720` + `depth_stab2.py … 3 0.035 0.06 1280 720` → `out/depth_hsil_t`, `out/depth_bal3_t`.
- Hands (`6bb37458-5823524`, dark on bright sky): `S_R=70 python3 sil_prep.py src out_relit.mp4` (sky fit → matte,
  rounded relief + side key + rim, on black), then both: `env $(cat dancer_preset_v6.env) OUTS=1.5 STILL=… python3
  depth_sheet_clip.py <depth> 0 50 x.mp4 <relit | footage> 0`.
- **Hands v2 (black hands, true to footage)**: no relight, no depth model (it snapped on the silhouette): flat synthetic
  depth `out/depth_hflat` (constant 0.5, 50 PNGs) + `env $(cat dancer_preset_v6.env) MATTE= TILT=0 YAW=0
  DARKCUT=0.28,0.45 DLOHI=0,1 FIT=cover PULSEK=0 OUTS=1.5 STILL=… python3 depth_sheet_clip.py out/depth_hflat 0 50 x.mp4 depthjobs/hsil_t.mp4 0`.
  New options: `DLOHI` (fixed depth range), `PULSEK` (beat pulse strength; 0 = off — the pulse pumped the full-frame sheet).

## 21 · Ballerina fix (`ef68eb02-ballerina_fix.mp4`, 377 fr) — 3 s test
- Test = clip 2.8–6.2 s (`ffmpeg -ss 2.8 -frames:v 85` → `depthjobs/balf_t.mp4`), render frames 5–79 (= 3.0–6.0 s).
- Depth at 1920: `depth_full2.py … 85 0 1920 1080` + `depth_stab2.py … 3 0.035 0.06 1920 1080` → `out/depth_balf_t`.
- Skin lift (footage only, depth from the ungraded clip): `lutrgb` each channel `255*pow(max(val/255-0.03,0)/0.97,0.6)` → `depthjobs/balf_g0.6.mp4`.
- `env $(cat dancer_preset_v6.env) PITCH=4.2 OUTS=1.5 RANGE=5,43 | 43,80 STILL=out/balf_test/balf python3 depth_sheet_clip.py out/depth_balf_t 0 85 x.mp4 depthjobs/balf_g0.6.mp4 0`
  (PITCH 6 → 4.2: ~9 → ~13 dots across the face; 4K: OUTS=3).
- **Ballerina fix full (12 s, clip frames 0–299)**: `ffmpeg -i <clip> -frames:v 300` → `depthjobs/balf_full.mp4`; same skin
  lift → `balf_full_g.mp4`; depth 1920 (`depth_full2` + `depth_stab2 … 3 0.035 0.06`) → `out/depth_balf`.
  `env $(cat presets/balf_full.env) OUTS=1.5 RANGE=0,150 | 150,300 STILL=out/balf_full/balf python3 depth_sheet_clip.py out/depth_balf 0 300 x.mp4 depthjobs/balf_full_g.mp4 0`
  New `TRANS_MODE=arc` (depth_sheet `arc_transition`): in along a Bezier arc (`ARC_S`/`ARC_C` offsets from her landing
  centroid, default 447,-407 / 325,-10 = from the top right, curving down then left), per-dot 3D helix (depth → brightness),
  fan-out spread, ~12% laggards, materialise over the first third; out along `ODIR` (default -0.985,-0.17) with lift `OLIFT`.

## 22 · Kiss (second chorus) — 2 s test
- `9ff6734f-couple_fix_2.mp4` (208 fr), test = frames 150–199 (6.0–8.0 s: leaning in → kiss).
- `env $(cat presets/dance_full.env) FOOT_CROP=230,0,1500 CLIPLEN=100 python3 motion_sil3.py $A/9ff6734f-couple_fix_2.mp4 0 208 out/kiss_test 150 175`
  (+ `175 200` in parallel). CLIPLEN=100 = no transitions in the test.

## 6b · Dancer v7 (more detail, brighter) — replaces v6
- Same source/depth/transitions as v6 (`TRANS=1 CLIPLEN=15 T0ABS=1`, default swirl; verified frame-for-frame vs v6).
- Skin lift on the footage: lutrgb `255*pow(max(val/255-0.03,0)/0.97,0.75)` → `depthjobs/dancer_g75.mp4` (0.6 + GAIN 1.5 blew the dress out).
- `env $(cat dancer_preset_v6.env) TRANS=1 CLIPLEN=15 T0ABS=1 PITCH=3.6 GAIN=1.0 OUTS=1.5 RANGE=0,188 | 188,375 STILL=out/dancer_v7/dancer python3 depth_sheet_clip.py out/depth_dancer_v3 0 375 x.mp4 depthjobs/dancer_g75.mp4 0`

## 23 · Reflections overlays (bokeh) — 2 s tests
- `38a55e86-reflections.mp4` (94 fr, test 0.8–2.8 s) and `66d8c8f1-reflections2.mp4` (290 fr, test 4–6 s), trimmed to `depthjobs/refl{1,2}_t.mp4`.
- Flat depth `out/depth_hflat`; `env $(cat dancer_preset_v6.env) ORBIT= MATTE= TILT=0 YAW=0 DLOHI=0,1 PULSEK=0 NOSPECKS=1 THIN=
  OUTS=1.5 FIT=cover DARKCUT=0.04,0.12` + refl1 `LUMSCALE=0.7 GAIN=0.9 TWK=4`, refl2 `LUMSCALE=0.8 GAIN=1.2 TWK=3`.
  New `LUMSCALE` (depth_sheet_clip): scales the footage light → dimmer + warmer (overlays).
- Transitions still to build (refs): refl1 in from top-right moving down-left, out down-left; refl2 in from the right, out to the left.
- refl1 v2 (brighter overlaps): footage pre-curve `lutrgb … 255*pow(val/255,1.5)` → `depthjobs/refl1_tc.mp4`, then `LUMSCALE=0.86 GAIN=0.95 TWK=4`.
- **Full runs**: flat depth extended to 290 frames; common `TRANS=1 TRANS_MODE=hands IN_XMIN=0 T0ABS=1 OUTK=1.0 OUTS=1.5` + the test settings.
  refl1 (`depthjobs/refl1_fc.mp4` = full clip with the 1.5 pre-curve, 94 fr): `FLOW_DIR=-0.7,0.7 ODIR=-0.81,0.59 TRANS_IN=1.6,1.6 TRANS_OUT=1.4,1.6 OUTPAD=0.1 CLIPLEN=3.76`.
  refl2 (290 fr): `FLOW_DIR=-0.99,0.12 ODIR=-1,0.05 TRANS_IN=2.6,2.4 TRANS_OUT=2.2,2.6 OUTPAD=0.25 CLIPLEN=11.6`, RANGE 0,145 | 145,290.
  New `ODIR` / `OSCAT` for TRANS_MODE=hands (directional exit).

## 24 · Birds flock (`66eaaad7-5434557`, 30 fps → frame-for-frame at 25, 240 fr) — 2 s test
- `birds_prep.py src out.mp4 [f0 n]`: black-top-hat key (41 px ellipse; dark vs local sky → cloud bands vanish) → lit
  birds on black. Test: `birds_prep.py <src> depthjobs/birds_t.mp4 95 50`.
- `env $(cat dancer_preset_v6.env) ORBIT= MATTE= TILT=0 YAW=0 DLOHI=0,1 PULSEK=0 NOSPECKS=1 THIN= OUTS=1.5 FIT=cover
  DARKCUT=0.04,0.12 PITCH=3.0 GAIN=1.4 TWK=3 STILL=… python3 depth_sheet_clip.py out/depth_hflat 0 50 x.mp4 depthjobs/birds_t.mp4 0`
  (gulls_full.py route = `birds_full.py` keyed the cloud bands; dropped). Transitions to build: in from the right, out to the left.

## 25 · Light-streak warp (`38a30124-12491717`, 30 fps → 25) — 2 s test
- Test = source frames 100–149 (`select=gte(n\,100),setpts=N/25/TB -r 25`), floor cut + curve:
  lutrgb `255*pow(max(val/255-0.075,0)/0.925,1.15)` → `depthjobs/warp_tc.mp4`.
- Same base as birds, `PITCH=3.6 DARKCUT=0.08,0.2 GAIN=1.1 TWK=3`. Transitions to build: radial burst out from the centre (in and out).
- **Warp v2 (overlay, softer)**: `warp_prep.py depthjobs/warp_t.mp4 depthjobs/warp_tv.mp4` (floor cut 0.075 + curve 1.15, then a
  per-ray brightness field: angle noise around the vanishing point 960,560, skewed W_GP=2.2 so most streaks are dim; core
  quieter), render `PITCH=3.6 DARKCUT=0.05,0.14 GAIN=0.9 LUMSCALE=0.6 TWK=3`.
- **Warp v3 (fainter)**: same plate, `DARKCUT=0.04,0.12 GAIN=0.6 LUMSCALE=0.45 TWK=2.5 HALO=0.3`.
- **Birds + warp full runs**: plates `birds_prep.py <src> depthjobs/birds_full.mp4` (240 fr) and
  `ffmpeg -vf setpts=N/25/TB -r 25` → `warp_prep.py depthjobs/warp_full.mp4 depthjobs/warp_fv.mp4` (300 fr); flat depth 300 fr.
  Common: `… FIT=cover TRANS=1 TRANS_MODE=hands IN_XMIN=0 T0ABS=1 OUTK=1.0 OUTPAD=0.25 TRANS_IN=2.6,2.4 TRANS_OUT=2.2,2.6 OUTS=1.5`.
  Birds: `DARKCUT=0.04,0.12 PITCH=3.0 GAIN=1.4 TWK=3 FLOW_DIR=-1,0 ODIR=-1,0 CLIPLEN=9.6 LUMDEP=0.05,0.2` (RANGE 0,120 | 120,240).
  Warp: `PITCH=3.6 DARKCUT=0.04,0.12 GAIN=0.6 LUMSCALE=0.45 TWK=2.5 HALO=0.3 IN_RADIAL=640,373 OCXY=640,373 CLIPLEN=12 LUMDEP=0.03,0.10` (RANGE 0,150 | 150,300).
  New: `IN_RADIAL=x,y` (hands mode: burst in from a point, inner first), `OCXY` (fixed dispersal centre), `LUMDEP=t0,t1`
  (flat plates: only lit dots fly / host sparks, via depth_sheet `MOVERMAP`; the look is unchanged).
- **Warp source fix**: `12491717` is 25 fps content with every 6th frame duplicated (300 → 250 unique). Plate rebuilt from the
  unique frames (drop frames whose diff to the previous < 0.25×median) → `depthjobs/warp_full.mp4` (250 fr, real time), then
  warp_prep; render with `CLIPLEN=10`, 0 250, RANGE 0,125 | 125,250. (Birds source checked: no duplicates.)

## 26 · Side screens: 9:16 rose loop (style frame stage)
- Model `Red_rose_SF.obj` (+ `.mtl`; maps uploaded as images: colour `a2ae3c42-image.jpg`, normal `50b48692-image.jpg`,
  occlusion `d4d39c86-image.jpg`, 2576², OBJ is Y-up). `common.py` now takes `DESIGN=720x1280` (9:16 canvas; OUTS=1.5 → 1080×1920).
- `env $(cat presets/rose916_sf.env) python3 rose916.py <obj> <colour> <normal> <ao> outdir f0 f1 [t1,t2,…]` (t = loop fractions
  → stills). Loop 493 frames (48 beats, 19.72 s), 4 turns (= previous rose speed); dissolve wave up the rose (TD0/TDS),
  particles spiral up and evaporate, rebuild wave (TR0/TRS) from twins rising from below; ripple ahead of the fronts;
  framing fitted to the bounding cylinder over the turn + the flight envelope (4 % / 3.5 % margins) → rose ≈ 66 % of the height.
  Style frames: t = 0.12, 0.33, 0.45, 0.55, 0.66. TILT=0 YAW=0 needed (the sheet's default camera shifted it off-centre).
