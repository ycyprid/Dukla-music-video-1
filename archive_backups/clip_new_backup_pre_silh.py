"""Motion tests for new footage in the locked halftone look (fixed screen grid, light from the footage).
usage: python3 clip_new.py src.mp4 depthdir start n mode out.mp4    mode: leaf | water | fireworks
depthdir holds stabilised 16-bit depth for frames start..start+n-1, named d0000.. (from the clip's own start)."""
import sys, os, glob
import numpy as np, cv2
import common
common.FPS = 25
import depth_sheet as DS
from common import W, H, Writer

src, dd, a, n, mode, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5], sys.argv[6]
cap = cv2.VideoCapture(src)
for _ in range(a):
    cap.grab()
frames = []
for _ in range(n):
    ok, fr = cap.read()
    h0, w0 = fr.shape[:2]
    if w0 / h0 > 16 / 9 + 0.01:                                    # centre-crop 2048x1080 -> 16:9
        cw = int(round(h0 * 16 / 9)); x0 = (w0 - cw) // 2; fr = fr[:, x0:x0 + cw]
    frames.append(cv2.resize(fr, (W, H), interpolation=cv2.INTER_AREA))
DOFF = int(os.environ.get('DEPTH_OFFSET', '0'))
deps = [cv2.resize(cv2.imread(f'{dd}/d{i + DOFF:04d}.png', -1).astype(np.float32) / 65535, (W, H)) for i in range(n)]
# one range for the whole clip (no pumping)
dlo = np.median([np.percentile(x, 2) for x in deps[::5]]); dhi = np.median([np.percentile(x, 99.7) for x in deps[::5]])
gs = [(f.max(2) if mode == 'fireworks' else cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)).astype(np.float32) / 255 for f in frames]
LREF = float(os.environ['LREF']) if os.environ.get('LREF') else max(np.percentile(np.stack(gs[::5]), 99.5), 0.2)
wr = Writer(out, None, [])
ST = {}
IDS = None
if os.environ.get('TRANS'):                      # surface transition (water): in from upstream, out downstream
    BT = 60.0 / 146; TL = float(os.environ.get('CLIPLEN', n / 25.0))
    fdx, fdy = (float(x_) for x_ in os.environ.get('FLOW_DIR', '-1,0').split(','))
    if os.environ.get('SOFT'):                     # ripple-paced: long sweep, gentle flights (16 beats in, 16 out)
        DS.TRANS.update(dict(mode='flow', kind='surface', soft=True, fdir=(fdx, fdy), ripple_wl=320.0, ripple_v=83.0,
                             in0=0.0, in_spread=5.5 * BT, in_dur=4.5 * BT,
                             out0=TL - 4.4, out_spread=5.5 * BT, out_dur=4.5 * BT))
    else:
        DS.TRANS.update(dict(mode='flow', kind='surface', fdir=(fdx, fdy),
                             in0=0.0, in_spread=2.2 * BT, in_dur=1.9 * BT,
                             out0=TL - 2.0, out_spread=2.2 * BT, out_dur=1.9 * BT))
    gx_, gy_ = DS.lattice(); IDS = np.arange(gx_.size, dtype=np.int64)   # fixed grid: stable identities
TRAIL = mode == 'leaf' and os.environ.get('TRAIL')
if TRAIL:
    # trailing sparkles: shed from the leaf with its velocity, slowed by air drag -> they fall behind along its path
    prng = np.random.default_rng(7)
    P_x = np.zeros(0); P_y = np.zeros(0); P_vx = np.zeros(0); P_vy = np.zeros(0); P_age = np.zeros(0)
    P_life = np.zeros(0); P_ph = np.zeros(0); P_fq = np.zeros(0); P_c = np.zeros(0); P_sw = np.zeros(0)
    prev_c = None; vel_s = np.zeros(2)
    SPAWN = int(os.environ.get('TRAIL_N', '34'))
for i in range(n):
    d = np.clip((deps[i] - dlo) / (dhi - dlo + 1e-6), 0, 1)
    g = gs[i]; lum = g / LREF
    if mode == 'leaf':
        lit = (cv2.GaussianBlur(g, (0, 0), 1.5) > 0.05).astype(np.uint8)
        lit = cv2.morphologyEx(lit, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
        mt = cv2.GaussianBlur(cv2.dilate(lit, np.ones((3, 3), np.uint8)).astype(np.float32), (0, 0), 1.2)
        dep = np.clip(d * mt / 0.85, 0, 1)
    elif mode == 'water':
        dep = 0.30 + 0.70 * d
    else:
        pres = np.clip(cv2.GaussianBlur(g, (0, 0), 1.2) / 0.08, 0, 1)
        if os.environ.get('SPARKMASK'):
            # keep sparks (sharp, bright, high-contrast) and their near glow; drop soft diffuse puffs (smoke / haze)
            hp_ = g - cv2.GaussianBlur(g, (0, 0), 3)
            keep = np.maximum(np.clip((hp_ - 0.02) / 0.04, 0, 1), np.clip((g - 0.62) / 0.2, 0, 1))
            if os.environ.get('SPARKMASK') == 'wide':
                keep = cv2.GaussianBlur(cv2.dilate(keep, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))), (0, 0), 1.5)
            else:                                        # tight: the sparks' own detail only (smoke can't ride along)
                keep = cv2.GaussianBlur(keep, (0, 0), 1.2)
            pres = pres * np.clip(keep * 1.4, 0, 1)
        dep = pres * (0.55 + 0.45 * cv2.GaussianBlur(d, (0, 0), 6))
    if TRAIL:
        dt = 1 / 25.0
        sel = mt > 0.5
        if sel.sum() > 30:
            ys_, xs_ = np.nonzero(sel); cc = np.array([xs_.mean(), ys_.mean()])
            if prev_c is not None:
                vel_s = 0.6 * vel_s + 0.4 * (cc - prev_c) / dt
            prev_c = cc
            # spawn: mostly from the edges (where the air peels them off), some from the surface
            edge = (cv2.morphologyEx(sel.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((5, 5), np.uint8)) > 0)
            ey, ex = np.nonzero(edge)
            k = SPAWN
            ke = int(k * 0.65)
            pe = prng.integers(0, len(ex), ke) if len(ex) else np.zeros(0, int)
            ps = prng.integers(0, len(xs_), k - ke)
            nx_ = np.concatenate([ex[pe], xs_[ps]]).astype(float) + prng.uniform(-1.5, 1.5, k)
            ny_ = np.concatenate([ey[pe], ys_[ps]]).astype(float) + prng.uniform(-1.5, 1.5, k)
            f_ = prng.uniform(0.55, 0.9, k)
            P_x = np.concatenate([P_x, nx_]); P_y = np.concatenate([P_y, ny_])
            P_vx = np.concatenate([P_vx, vel_s[0] * f_ + prng.normal(0, 18, k)]); P_vy = np.concatenate([P_vy, vel_s[1] * f_ + prng.normal(0, 18, k)])
            P_age = np.concatenate([P_age, np.zeros(k)]); P_life = np.concatenate([P_life, prng.uniform(0.55, 1.25, k)])
            P_ph = np.concatenate([P_ph, prng.random(k)]); P_fq = np.concatenate([P_fq, prng.uniform(3, 8, k)])
            P_c = np.concatenate([P_c, prng.uniform(200, 252, k)]); P_sw = np.concatenate([P_sw, prng.uniform(-1, 1, k)])
        # air drag + a little gravity + a gentle curl
        dr_ = np.exp(-1.8 * dt)
        P_vx *= dr_; P_vy = P_vy * dr_ + 30 * dt
        P_x += (P_vx + 22 * P_sw * np.sin(2 * np.pi * (0.9 * P_age + P_ph))) * dt; P_y += P_vy * dt
        P_age += dt
        keep = P_age < P_life
        P_x, P_y, P_vx, P_vy, P_age, P_life, P_ph, P_fq, P_c, P_sw = (q[keep] for q in (P_x, P_y, P_vx, P_vy, P_age, P_life, P_ph, P_fq, P_c, P_sw))
        a_ = P_age / P_life
        env = (1 - a_) ** 1.5 * (1 - np.exp(-P_age / 0.05)) * (0.45 + 0.55 * np.maximum(np.sin(2 * np.pi * (P_fq * P_age + P_ph)), 0) ** 4)
        DS.EXTRA.update(u=P_x, v=P_y, w=env * float(os.environ.get('TRAIL_K', '1.6')), col=DS.L.PAL[P_c.astype(np.uint8)])
    wr.write(DS.render(dep.astype(np.float32), 1, (a + i) / 25.0, None, None, IDS, ST, lum.astype(np.float32)))
wr.close()
print('done', out)
