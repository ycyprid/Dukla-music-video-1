"""Style frames for new footage: depth from ONE frame (Depth Anything V2 small) + the locked halftone look.
usage: python3 stills_new.py video frame_index mode out.png     mode: leaf | water | fireworks
(look settings come from the environment, as for the other shots)"""
import sys, os
import numpy as np, cv2, onnxruntime as ort
import depth_sheet as DS
from common import W, H

src, fi, mode, out = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
cap = cv2.VideoCapture(src)
for _ in range(fi):
    cap.grab()
ok, fr = cap.read()
# centre-crop wider-than-16:9 footage (2048x1080 -> 1920x1080)
h0, w0 = fr.shape[:2]
if w0 / h0 > 16 / 9 + 0.01:
    cw = int(round(h0 * 16 / 9)); x0 = (w0 - cw) // 2; fr = fr[:, x0:x0 + cw]
frw = cv2.resize(fr, (W, H), interpolation=cv2.INTER_AREA)
cv2.imwrite(out.replace('.png', '_src.jpg'), frw)
# ---- depth for this single frame ----
so = ort.SessionOptions(); so.intra_op_num_threads = 2
sess = ort.InferenceSession('models/depth_anything_v2_vits_dynamic.onnx', so, providers=['CPUExecutionProvider'])
IW, IH = 518, 294
mean = np.array([0.485, 0.456, 0.406], np.float32); std = np.array([0.229, 0.224, 0.225], np.float32)
x = cv2.resize(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB), (IW, IH), interpolation=cv2.INTER_CUBIC).astype(np.float32) / 255
x = ((x - mean) / std).transpose(2, 0, 1)[None]
d = sess.run(None, {sess.get_inputs()[0].name: x})[0].squeeze().astype(np.float32)
d = cv2.resize(d, (W, H), interpolation=cv2.INTER_CUBIC)
d = np.clip((d - np.percentile(d, 2)) / (np.percentile(d, 99.7) - np.percentile(d, 2) + 1e-6), 0, 1)
cv2.imwrite(out.replace('.png', '_depth.png'), (d * 65535).astype(np.uint16))
g = (frw.max(2) if mode == 'fireworks' else cv2.cvtColor(frw, cv2.COLOR_BGR2GRAY)).astype(np.float32) / 255   # sparks: brightest channel (blue counts)
LREF = max(np.percentile(g, 99.5), 0.2)
lum = g / LREF
if mode == 'leaf':
    # subject on black: tight cut-out from the footage (lit or right next to lit), depth for relief
    lit = (cv2.GaussianBlur(g, (0, 0), 1.5) > 0.05).astype(np.uint8)
    lit = cv2.morphologyEx(lit, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    mt = cv2.GaussianBlur(cv2.dilate(lit, np.ones((3, 3), np.uint8)).astype(np.float32), (0, 0), 1.2)
    dd = d * mt
    sel = mt > 0.5
    dep = np.clip(dd / (np.percentile(dd[sel], 95) + 1e-6) * 0.8, 0, 1) if sel.any() else dd
elif mode == 'water':
    # the whole frame is surface: everything present, depth = the receding plane + ripple relief
    dep = 0.30 + 0.70 * d
elif mode == 'scene':
    # full-frame scene (foliage, sky): everything present, light from the footage, depth for relief
    dep = 0.30 + 0.70 * d
    DW = float(os.environ.get('DEPTH_LIGHT', '0'))              # >0: nearer = brighter (the hero pops, the far falls back)
    if DW > 0:
        lum = lum * ((1 - DW) + DW * np.clip(cv2.GaussianBlur(d, (0, 0), 3) / 0.8, 0, 1.15))
elif mode == 'silh':
    # backlit silhouettes against sky: the plants are the subject, the sky goes to black.
    # presence = darker than the sky level; light = backlit rim at the sky edge + the leaves' own texture
    gb = cv2.GaussianBlur(g, (0, 0), 1.0)
    # smooth sky model: quadratic fit to bright, flat pixels (handles a sky that darkens across the frame)
    sm = cv2.resize(gb, (192, 108), interpolation=cv2.INTER_AREA)
    det = np.abs(sm - cv2.GaussianBlur(sm, (0, 0), 2))
    cand = (sm > np.percentile(sm, 60) * 0.9) & (det < 0.04) & (sm > 0.25)
    yy, xx = np.mgrid[0:108, 0:192].astype(np.float32); xx /= 192; yy /= 108
    Bs = np.stack([np.ones_like(xx), xx, yy, xx * xx, xx * yy, yy * yy], -1)
    coef = np.linalg.lstsq(Bs[cand], sm[cand], rcond=None)[0]
    skylo = 0.75 * np.median(sm[cand]); sky = cv2.resize(np.clip(Bs @ coef, skylo, 1.0).astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)
    T = sky * float(os.environ.get('SILH_T', '0.72'))
    m = np.clip((T - gb) / (0.12 * sky), 0, 1)
    m = cv2.GaussianBlur(m, (0, 0), 1.0)
    edge = np.clip(cv2.GaussianBlur(np.abs(cv2.Laplacian(cv2.GaussianBlur(m, (0, 0), 1.5), cv2.CV_32F)), (0, 0), 1.5) * 9, 0, 1)
    rim = edge * m ** 0.3
    hp = np.abs(gb - cv2.GaussianBlur(gb, (0, 0), 4)) / sky
    tex = np.clip(hp * 5, 0, 1)
    own = np.clip(gb / (0.6 * sky), 0, 1) ** 0.9                       # the subject's own light, relative to the sky
    lum = np.clip(m * (float(os.environ.get('SILH_IN', '0.6')) * own + float(os.environ.get('SILH_TEX', '0.45')) * tex)
                  + float(os.environ.get('SILH_RIM', '1.1')) * rim, 0, 1.2)
    dep = m * (0.45 + 0.55 * d)
    LREF = 1.0
elif mode == 'depthsubj':
    # pale subject against a light sky: depth says what is subject (sky -> black), the subject keeps its own light
    import mattes
    g8 = cv2.cvtColor(cv2.resize(fr, (640, 360), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    mm, ab = mattes.depth_matte(d, g8, lift=float(os.environ.get('DS_LIFT', '0.07')))
    # fine edge from colour: distance to a smooth sky-colour model fitted where depth says sky
    f32 = frw.astype(np.float32) / 255
    sm_ = cv2.resize(f32, (160, 90), interpolation=cv2.INTER_AREA); msk = cv2.resize(mm, (160, 90)) < 0.05
    yy, xx = np.mgrid[0:90, 0:160].astype(np.float32); xx = xx / 160 - 0.5; yy = yy / 90 - 0.5
    Bq = np.stack([np.ones_like(xx), xx, yy, xx * xx, xx * yy, yy * yy], -1)
    skyc = np.stack([cv2.resize((Bq @ np.linalg.lstsq(Bq[msk], sm_[..., c][msk], rcond=None)[0]).astype(np.float32), (W, H),
                                interpolation=cv2.INTER_CUBIC) for c in range(3)], -1)
    dist = np.sqrt(((cv2.GaussianBlur(f32, (0, 0), 0.8) - skyc) ** 2).sum(-1))
    c0, c1 = (float(x_) for x_ in os.environ.get('SKYKEY', '0.05,0.13').split(','))
    al = np.clip((dist - c0) / (c1 - c0), 0, 1); al = al * al * (3 - 2 * al)
    near_ = cv2.GaussianBlur(cv2.dilate(mm, np.ones((21, 21), np.uint8)), (0, 0), 4)
    mm = al * np.clip(near_, 0, 1)
    sel = mm > 0.5
    ref = np.percentile(g[sel], float(os.environ.get('DS_REFP', '93'))) if sel.sum() > 100 else 1.0
    lum = np.clip(g / ref, 0, 1.2) * mm
    BD = float(os.environ.get('BRANCH_DETAIL', '0'))
    if BD > 0:
        # the branch is a flat dark silhouette in the footage: give it form from its own shape --
        # rounded (cylinder) shading lit from the upper left, a thin backlit rim, boosted micro-contrast
        br = mm * np.clip((0.42 - g / ref) / 0.22, 0, 1)
        bb = (br > 0.5).astype(np.uint8)
        dt = cv2.distanceTransform(bb, cv2.DIST_L2, 5)
        hw = cv2.GaussianBlur(cv2.dilate(dt, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))), (0, 0), 4) + 1e-3
        r = np.clip(dt / hw, 0, 1)                                          # 0 at the edge .. 1 on the spine
        gy_, gx_ = np.gradient(cv2.GaussianBlur(dt, (0, 0), 1.5))
        gn = np.sqrt(gx_ ** 2 + gy_ ** 2) + 1e-6
        side = 1 - r
        nx, ny = -gx_ / gn * side, -gy_ / gn * side; nz = np.sqrt(np.clip(1 - side ** 2, 0, 1))
        Lb = np.array([-0.55, -0.6, 0.58]); Lb /= np.linalg.norm(Lb)
        shade = np.clip(nx * Lb[0] + ny * Lb[1] + nz * Lb[2], 0, 1)
        rimb = side ** 5 * bb
        g8_ = cv2.cvtColor(frw, cv2.COLOR_BGR2GRAY)
        cl = cv2.createCLAHE(4.0, (16, 16)).apply(g8_).astype(np.float32) / 255
        micro = np.clip((cl - cv2.GaussianBlur(cl, (0, 0), 3)) * 4 + 0.5, 0, 1)
        blum = (0.06 + float(os.environ.get("BRANCH_SHADE", "0.42")) * shade ** 1.6) * (0.75 + 0.5 * micro) + float(os.environ.get("BRANCH_RIM", "0.55")) * rimb
        bs = cv2.GaussianBlur(br, (0, 0), 1.0) * BD
        lum = lum * (1 - bs) + blum * bs
    LREF = 1.0
    dep = mm * (0.45 + 0.55 * d)
elif mode == 'snow':
    # every flake is its own particle, shaped by its own streak; no dot sheet
    gfu = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255     # full res: flakes are tiny
    hp = gfu - cv2.GaussianBlur(gfu, (0, 0), 4)
    bw = (hp > float(os.environ.get('SNOW_T', '0.05'))).astype(np.uint8)
    n_, lab, st_, cen = cv2.connectedComponentsWithStats(bw, 8)
    U, V, Wt, Ci = [], [], [], []
    rng_ = np.random.default_rng(5)
    for k in range(1, n_):
        a_ = st_[k, cv2.CC_STAT_AREA]
        if a_ < 2:
            continue
        x0, y0, w0, h0 = st_[k, :4]
        sub = (lab[y0:y0 + h0, x0:x0 + w0] == k) * hp[y0:y0 + h0, x0:x0 + w0]
        flux = sub.sum(); yy_, xx_ = np.nonzero(sub)
        ww = sub[yy_, xx_]; cx_ = (xx_ * ww).sum() / ww.sum() + x0; cy_ = (yy_ * ww).sum() / ww.sum() + y0
        if len(xx_) > 2:
            cov = np.cov(np.stack([xx_, yy_]), aweights=ww + 1e-6); ev, evec = np.linalg.eigh(cov)
            L = 2.2 * np.sqrt(max(ev[1], 0.05)); ax = evec[:, 1]
        else:
            L = 0.5; ax = np.array([1.0, 0.0])
        npt = int(np.clip(L / 0.8, 1, 14))
        ts = np.linspace(-1, 1, npt) if npt > 1 else np.zeros(1)
        prof = (1 - 0.6 * ts ** 2)
        wpt = flux ** 0.75 * prof / prof.sum()
        U.append(cx_ + ax[0] * ts * L * 0.5); V.append(cy_ + ax[1] * ts * L * 0.5); Wt.append(wpt)
        ci_ = rng_.uniform(236, 255) if rng_.random() < 0.75 else rng_.uniform(200, 232)
        Ci.append(np.full(npt, ci_))
    U = np.concatenate(U) * (W / fr.shape[1]); V = np.concatenate(V) * (H / fr.shape[0])
    Wt = np.concatenate(Wt) * float(os.environ.get('SNOW_K', '1.0')); Ci = np.concatenate(Ci)
    print('flakes', n_ - 1, 'particles', U.size)
    DS.EXTRA.update(u=U, v=V, w=Wt, col=DS.L.PAL[Ci.astype(np.uint8)])
    dep = np.zeros_like(g); lum = np.zeros_like(g); LREF = 1.0
elif mode == 'fireworks':
    # sparks in the sky: their own light says where the dots are; depth only adds a faint relief
    pres = np.clip(cv2.GaussianBlur(g, (0, 0), 1.2) / 0.08, 0, 1)
    dep = pres * (0.55 + 0.45 * cv2.GaussianBlur(d, (0, 0), 6))
else:
    raise SystemExit('mode?')
img = DS.render(dep.astype(np.float32), 1, fi / 25.0, None, None, None, None, lum.astype(np.float32))
cv2.imwrite(out, (np.clip(img, 0, 1) * 65535 + 0.5).astype(np.uint16))
print('done', out)
