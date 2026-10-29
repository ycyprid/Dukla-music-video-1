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
frames = []; SNOWF = []
for _ in range(n):
    ok, fr = cap.read()
    h0, w0 = fr.shape[:2]
    if w0 / h0 > 16 / 9 + 0.01:                                    # centre-crop 2048x1080 -> 16:9
        cw = int(round(h0 * 16 / 9)); x0 = (w0 - cw) // 2; fr = fr[:, x0:x0 + cw]
    if mode == 'snow':                               # snow needs only the full-res luma (memory)
        SNOWF.append(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)); frames.append(np.zeros((H, W, 3), np.uint8)) if not frames else frames.append(frames[0])
    else:
        frames.append(cv2.resize(fr, (W, H), interpolation=cv2.INTER_AREA))
DOFF = int(os.environ.get('DEPTH_OFFSET', '0'))
def ldep(i):                                       # depth loaded per frame (a full clip in memory OOMs)
    if dd == 'none':
        return np.full((H, W), 0.5, np.float32)
    return cv2.resize(cv2.imread(f'{dd}/d{i + DOFF:04d}.png', -1).astype(np.float32) / 65535, (W, H))
# one range for the whole clip (no pumping)
_smp = [ldep(i) for i in range(0, n, 5)]
dlo = np.median([np.percentile(x, 2) for x in _smp]); dhi = np.median([np.percentile(x, 99.7) for x in _smp]); del _smp
gs = [(f.max(2) if mode == 'fireworks' else cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)) for f in frames]   # uint8 (memory)
LREF = float(os.environ['LREF']) if os.environ.get('LREF') else max(np.percentile(np.stack(gs[::5]), 99.5) / 255, 0.2)
wr = Writer(out, None, [])
ST = {}; SK = {}
IDS = None
if os.environ.get('TRANS'):                      # surface transition (water): in from upstream, out downstream
    BT = 60.0 / 146; TL = float(os.environ.get('CLIPLEN', n / 25.0))
    fdx, fdy = (float(x_) for x_ in os.environ.get('FLOW_DIR', '-1,0').split(','))
    if os.environ.get('TRANS_MODE') == 'hands':      # stream in along the shot's motion, disperse in all directions
        isp, idu = (float(x_) for x_ in os.environ.get('TRANS_IN', '2.6,2.4').split(','))
        osp, odu = (float(x_) for x_ in os.environ.get('TRANS_OUT', '2.2,2.6').split(','))
        ocx, ocy = (float(x_) for x_ in os.environ.get('OUT_C', '%f,%f' % (W / 2, H / 2)).split(','))
        DS.TRANS.update(dict(mode='hands', fly_soft=float(os.environ.get('FLY_SOFT', '0.8')), fdir=(fdx, fdy), in_xmin=0.0,
                             in0=0.0, in_spread=isp * BT, in_dur=idu * BT,
                             out0=TL - (osp * BT + float(os.environ.get('OUTK', '1.0')) * odu * BT) - float(os.environ.get('OUTPAD', '0.25')),
                             out_spread=osp * BT, out_dur=odu * BT, ocx=ocx, ocy=ocy))
    elif os.environ.get('SOFT'):
        SK_ = float(os.environ.get('SOFT_K', '1'))       # shorter shots: both transitions scaled                     # ripple-paced: long sweep, gentle flights (16 beats in, 16 out)
        DS.TRANS.update(dict(mode='flow', kind='surface', soft=True, fdir=(fdx, fdy), ripple_wl=320.0, ripple_v=83.0,
                             in0=0.0, in_spread=5.5 * BT * SK_, in_dur=4.5 * BT * SK_,
                             out0=TL - 4.4 * SK_, out_spread=5.5 * BT * SK_, out_dur=4.5 * BT * SK_))
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
    d = np.clip((ldep(i) - dlo) / (dhi - dlo + 1e-6), 0, 1)
    g = gs[i].astype(np.float32) / 255; lum = g / LREF
    if mode == 'leaf':
        lit = (cv2.GaussianBlur(g, (0, 0), 1.5) > 0.05).astype(np.uint8)
        lit = cv2.morphologyEx(lit, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
        mt = cv2.GaussianBlur(cv2.dilate(lit, np.ones((3, 3), np.uint8)).astype(np.float32), (0, 0), 1.2)
        dep = np.clip(d * mt / 0.85, 0, 1)
    elif mode in ('water', 'scene'):
        dep = 0.30 + 0.70 * d
        DW = float(os.environ.get('DEPTH_LIGHT', '0'))          # nearer = brighter (the hero pops, the far falls back)
        if DW > 0:
            lum = lum * ((1 - DW) + DW * np.clip(cv2.GaussianBlur(d, (0, 0), 3) / 0.8, 0, 1.15))
    elif mode == 'snow':
        # every flake its own particle, shaped by its own streak (full-res detection); faint flakes fade, never pop
        gfu = SNOWF[i].astype(np.float32) / 255
        hp = gfu - cv2.GaussianBlur(gfu, (0, 0), 4)
        T0 = float(os.environ.get('SNOW_T', '0.025'))
        bw = (hp > T0 * 0.7).astype(np.uint8)
        n_, lab, st_, _ = cv2.connectedComponentsWithStats(bw, 8)
        U, V, Wt, Ci = [], [], [], []
        for k in range(1, n_):
            if st_[k, cv2.CC_STAT_AREA] < 2:
                continue
            x0, y0, w0, h0 = st_[k, :4]
            sub = (lab[y0:y0 + h0, x0:x0 + w0] == k) * hp[y0:y0 + h0, x0:x0 + w0]
            pk = sub.max(); flux = sub.sum(); yy_, xx_ = np.nonzero(sub); ww = sub[yy_, xx_]
            cx_ = (xx_ * ww).sum() / ww.sum() + x0; cy_ = (yy_ * ww).sum() / ww.sum() + y0
            if len(xx_) > 2:
                cov = np.cov(np.stack([xx_, yy_]), aweights=ww + 1e-6); ev, evec = np.linalg.eigh(cov)
                Lf = 2.2 * np.sqrt(max(ev[1], 0.05)); ax = evec[:, 1]
            else:
                Lf = 0.5; ax = np.array([1.0, 0.0])
            npt = int(np.clip(Lf / 0.8, 1, 14)); ts = np.linspace(-1, 1, npt) if npt > 1 else np.zeros(1)
            prof = 1 - 0.6 * ts ** 2
            soft = np.clip((pk - T0 * 0.7) / (T0 * 0.8), 0, 1)
            U.append(cx_ + ax[0] * ts * Lf * 0.5); V.append(cy_ + ax[1] * ts * Lf * 0.5); Wt.append(flux ** 0.75 * prof / prof.sum() * soft)
            hsh = (int(cx_ * 7.3) * 73856093 ^ int(cy_ * 3.1) * 19349663) % 1000 / 1000.0    # colour: silver mostly, some gold
            Ci.append(np.full(npt, 236 + 19 * hsh if hsh < 0.75 else 200 + 32 * hsh))
        if U:
            sc_ = W / SNOWF[i].shape[1]
            DS.EXTRA.update(u=np.concatenate(U) * sc_, v=np.concatenate(V) * sc_, w=np.concatenate(Wt) * float(os.environ.get('SNOW_K', '2.6')),
                            col=DS.L.PAL[np.concatenate(Ci).astype(np.uint8)])
        else:
            DS.EXTRA.clear()
        dep = np.zeros_like(g); lum = np.zeros_like(g)
    elif mode == 'depthsubj':
        # pale subject against a light sky: depth says what is subject, colour gives the fine edge, sky -> black;
        # the flat dark branch gets form from its own shape (rounded shading, backlit rim, micro-contrast)
        import mattes
        frw = frames[i]
        g8 = cv2.resize(gs[i], (640, 360), interpolation=cv2.INTER_AREA)
        mm, ab = mattes.depth_matte(d, g8, lift=float(os.environ.get('DS_LIFT', '0.07')))
        f32 = frw.astype(np.float32) / 255
        sm_ = cv2.resize(f32, (160, 90), interpolation=cv2.INTER_AREA); msk = cv2.resize(mm, (160, 90)) < 0.05
        yq, xq = np.mgrid[0:90, 0:160].astype(np.float32); xq = xq / 160 - 0.5; yq = yq / 90 - 0.5
        Bq = np.stack([np.ones_like(xq), xq, yq, xq * xq, xq * yq, yq * yq], -1)
        cq = np.stack([np.linalg.lstsq(Bq[msk], sm_[..., c][msk], rcond=None)[0] for c in range(3)])
        if 'cq' in SK:
            cq = 0.75 * SK['cq'] + 0.25 * cq                         # sky model eased over time (no flicker)
        SK['cq'] = cq
        skyc = np.stack([cv2.resize((Bq @ cq[c]).astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC) for c in range(3)], -1)
        dist = np.sqrt(((cv2.GaussianBlur(f32, (0, 0), 0.8) - skyc) ** 2).sum(-1))
        c0, c1 = (float(x_) for x_ in os.environ.get('SKYKEY', '0.05,0.13').split(','))
        al = np.clip((dist - c0) / (c1 - c0), 0, 1); al = al * al * (3 - 2 * al)
        near_ = cv2.GaussianBlur(cv2.dilate(mm, np.ones((21, 21), np.uint8)), (0, 0), 4)
        mm = al * np.clip(near_, 0, 1)
        if os.environ.get('MATTE_DIR'):              # precomputed temporally consistent matte (blossom_matte.py)
            mm = cv2.imread(f"{os.environ['MATTE_DIR']}/m{a + i:04d}.png", 0).astype(np.float32) / 255
        sel = mm > 0.5
        ref = np.percentile(g[sel], float(os.environ.get('DS_REFP', '80'))) if sel.sum() > 100 else SK.get('ref', 0.6)
        if 'ref' in SK:
            ref = 0.9 * SK['ref'] + 0.1 * ref                         # exposure eased (no pumping)
        SK['ref'] = ref
        lum = np.clip(g / ref, 0, 1.2) * mm
        if os.environ.get('DBG'):
            print('DBG', i, 'mm %.4f al %.4f near %.4f ref %.3f sky %.3f' % (mm.mean(), al.mean(), near_.mean(), ref, skyc.mean()), flush=True)
        if float(os.environ.get('BRANCH_DETAIL', '0')) > 0:
            br = mm * np.clip((0.42 - g / ref) / 0.22, 0, 1)
            bb = (br > 0.5).astype(np.uint8)
            dt = cv2.distanceTransform(bb, cv2.DIST_L2, 5)
            hw = cv2.GaussianBlur(cv2.dilate(dt, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))), (0, 0), 4) + 1e-3
            rr_ = np.clip(dt / hw, 0, 1)
            gy_, gx_ = np.gradient(cv2.GaussianBlur(dt, (0, 0), 1.5)); gn = np.sqrt(gx_ ** 2 + gy_ ** 2) + 1e-6
            side = 1 - rr_
            nx, ny = -gx_ / gn * side, -gy_ / gn * side; nz = np.sqrt(np.clip(1 - side ** 2, 0, 1))
            Lb = np.array([-0.55, -0.6, 0.58]); Lb /= np.linalg.norm(Lb)
            shade = np.clip(nx * Lb[0] + ny * Lb[1] + nz * Lb[2], 0, 1)
            rimb = side ** 5 * bb
            cl = cv2.createCLAHE(4.0, (16, 16)).apply(gs[i]).astype(np.float32) / 255
            micro = np.clip((cl - cv2.GaussianBlur(cl, (0, 0), 3)) * 4 + 0.5, 0, 1)
            blum = (0.06 + float(os.environ.get('BRANCH_SHADE', '0.62')) * shade ** 1.6) * (0.75 + 0.5 * micro) \
                + float(os.environ.get('BRANCH_RIM', '0.7')) * rimb
            bs = cv2.GaussianBlur(br, (0, 0), 1.0)
            lum = lum * (1 - bs) + blum * bs
        dep = mm * (0.45 + 0.55 * d)
    elif mode == 'silh':
        # backlit silhouettes: smooth sky model (quadratic, eased over time so it can't flicker) -> sky to black,
        # plants carry a rim light at the sky edge + their own light and texture
        gb = cv2.GaussianBlur(g, (0, 0), 1.0)
        sm = cv2.resize(gb, (192, 108), interpolation=cv2.INTER_AREA)
        det = np.abs(sm - cv2.GaussianBlur(sm, (0, 0), 2))
        cand = (sm > np.percentile(sm, 60) * 0.9) & (det < 0.04) & (sm > 0.25)
        yy, xx = np.mgrid[0:108, 0:192].astype(np.float32); xx /= 192; yy /= 108
        Bs = np.stack([np.ones_like(xx), xx, yy, xx * xx, xx * yy, yy * yy], -1)
        coef = np.linalg.lstsq(Bs[cand], sm[cand], rcond=None)[0]; skylo_ = 0.75 * np.median(sm[cand])
        if 'coef' in SK:
            coef = 0.8 * SK['coef'] + 0.2 * coef; skylo_ = 0.8 * SK['skylo'] + 0.2 * skylo_
        SK['coef'] = coef; SK['skylo'] = skylo_
        sky = cv2.resize(np.clip(Bs @ coef, skylo_, 1.0).astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)
        T = sky * float(os.environ.get('SILH_T', '0.72'))
        m = cv2.GaussianBlur(np.clip((T - gb) / (0.12 * sky), 0, 1), (0, 0), 1.0)
        edge = np.clip(cv2.GaussianBlur(np.abs(cv2.Laplacian(cv2.GaussianBlur(m, (0, 0), 1.5), cv2.CV_32F)), (0, 0), 1.5) * 9, 0, 1)
        rim = edge * m ** 0.3
        tex = np.clip(np.abs(gb - cv2.GaussianBlur(gb, (0, 0), 4)) / sky * 5, 0, 1)
        own = np.clip(gb / (0.6 * sky), 0, 1) ** 0.9
        lum = np.clip(m * (float(os.environ.get('SILH_IN', '0.6')) * own + float(os.environ.get('SILH_TEX', '0.45')) * tex)
                      + float(os.environ.get('SILH_RIM', '1.1')) * rim, 0, 1.2)
        dep = m * (0.45 + 0.55 * d)
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
