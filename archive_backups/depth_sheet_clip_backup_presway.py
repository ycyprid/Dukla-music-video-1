"""2 s motion test of the depth dot-sheet look; dots ride on the footage (optical flow advection).
usage: python3 depth_sheet_clip.py depthdir start n out.mp4 src.mp4 src_offset"""
import sys, os, glob, numpy as np, cv2
import depth_sheet as DS
import mattes
from common import Writer, W, H
import common
common.FPS = 25
dd, a, n, out, src, off = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5], int(sys.argv[6])
fs = sorted(glob.glob(f'{dd}/d*.png'))[a:a + n]
# source frames (grayscale, 640x360) for flow
cap = cv2.VideoCapture(src)
for _ in range(off + a):
    cap.grab()                                   # frame-accurate (no keyframe seek)
gray = []; lumf = []
for _ in range(n):
    ok, fr = cap.read()
    lumf.append(cv2.cvtColor(cv2.resize(fr, (W, H), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY))
    gray.append(cv2.cvtColor(cv2.resize(fr, (640, 360), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY))
def matte(g8):
    g = cv2.resize(g8, (W, H)).astype(np.float32) / 255
    mt = (cv2.GaussianBlur(g, (0, 0), 2) > 0.11).astype(np.uint8)              # visible figure only (no faint trails)
    mt = cv2.morphologyEx(mt, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (45, 45)))  # keep dark bodice
    mt = cv2.dilate(mt, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))).astype(np.float32)
    return cv2.GaussianBlur(mt, (0, 0), 2.5)


MT = os.environ.get('MATTE')
smp = []
for j in range(0, n, 5):
    x = cv2.resize(cv2.imread(fs[j], -1).astype(np.float32) / 65535, (W, H))
    if MT == 'depth':
        mm, ab = mattes.depth_matte(x, gray[j]); x = ab * mm
    elif MT:
        x = x * matte(gray[j])
    smp.append(cv2.resize(x, (320, 180)))
lohi = (np.median([np.percentile(x, 2) for x in smp]), np.median([np.percentile(x, 99.7) for x in smp]))
LEVEL = None
if MT == 'depth':
    # one level per frame (her figure's 95th percentile), smoothed over time -> no pumping, no trails
    lv = []
    for j in range(n):
        x = cv2.resize(cv2.imread(fs[j], -1).astype(np.float32) / 65535, (W // 2, H // 2))
        mm, ab = mattes.depth_matte(x, cv2.resize(gray[j], (W // 2, H // 2)))
        sel = mm > 0.5
        lv.append((np.median(ab[sel]), np.percentile(ab[sel], 95)) if sel.sum() > 50 else (np.nan, np.nan))
    lv = np.array(lv)
    k = np.exp(-0.5 * (np.arange(-12, 13) / 4.0) ** 2); k /= k.sum()
    RAWLV, LEVEL = [], []
    for q in range(2):
        v = lv[:, q]; ok_ = ~np.isnan(v); v = np.interp(np.arange(n), np.nonzero(ok_)[0], v[ok_])
        RAWLV.append(v); LEVEL.append(np.convolve(np.pad(v, 12, mode='edge'), k, mode='valid'))
    RAWLV, LEVEL = np.array(RAWLV), np.array(LEVEL)          # [median, p95] per frame: measured / smoothed
    lohi = (0.0, float(np.median(LEVEL[1])))
dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
X0, Y0, LV, SHP = DS.lattice(levels=True)
pts = (X0, Y0)
scale = np.ones(1, np.float32)            # accumulated zoom of the footage
STILL = os.environ.get('STILL')                   # write 16-bit PNG stills (any output scale) instead of a video
wr = None if STILL else Writer(out, None, [])
TABS = os.environ.get('T0ABS')                    # time measured from the start of the clip (matches the full render)
ST = {} if os.environ.get('DOTSMOOTH', '1') == '1' else None
USELUM = os.environ.get('USELUM')
LREF = np.percentile(np.stack(lumf[::5]), 99.5) if USELUM else 1.0
NOTRACK = os.environ.get('NOTRACK')
if os.environ.get('ORBIT'):                      # specks orbit her: centre = her (smoothed) lit centroid
    cxs, cys = [], []
    for j in range(n):
        ys_, xs_ = np.nonzero(lumf[j] > 25)
        cxs.append(xs_.mean() if len(xs_) else W / 2); cys.append(ys_.mean() if len(ys_) else H / 2)
    kk = np.exp(-0.5 * (np.arange(-15, 16) / 6.0) ** 2); kk /= kk.sum()
    CXS = np.convolve(np.pad(cxs, 15, mode='edge'), kk, mode='valid'); CYS = np.convolve(np.pad(cys, 15, mode='edge'), kk, mode='valid')
for i, f in enumerate(fs):
    dep = DS.load_depth(f, lohi)
    if MT:                                       # subject on true black: keep only the figure
        dep = cv2.resize(cv2.imread(f, -1).astype(np.float32) / 65535, (W, H))
        if MT == 'depth':
            mm, ab = mattes.depth_matte(dep, gray[i])
            m0, q0 = RAWLV[:, i]; m1, q1 = LEVEL[:, i]                 # remap this frame's level/spread to the smoothed ones
            ab = np.clip((ab - m0) * (q1 - m1) / max(q0 - m0, 1e-3) + m1, 0, 1)
            if os.environ.get('THIN'):
                # slim cut-out: the depth matte only where the footage confirms the figure (lit, or right next to lit)
                lit = (cv2.GaussianBlur(lumf[i].astype(np.float32) / 255, (0, 0), 1.5) > 0.06).astype(np.uint8)
                lit = cv2.morphologyEx(lit, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)))
                lit = cv2.GaussianBlur(cv2.dilate(lit, np.ones((3, 3), np.uint8)).astype(np.float32), (0, 0), 1.2)
                mm = cv2.erode(mm, np.ones((5, 5), np.uint8)) * lit
            dep = ab * mm
        else:
            dep = dep * matte(gray[i])
        dep = np.clip((dep - lohi[0]) / (lohi[1] - lohi[0] + 1e-6), 0, 1)
    lm = (lumf[i].astype(np.float32) / LREF) if USELUM else None
    if os.environ.get('ORBIT'):
        DS.ORBIT['cx'] = float(CXS[i]); DS.ORBIT['cy'] = float(CYS[i]) * 0.5 + H * 0.25
        if os.environ.get('FIT'):                  # orbit / swirl axis in the fitted screen frame too
            fx, fy = DS.fit_frame(np.array([DS.ORBIT['cx']]), np.array([DS.ORBIT['cy']]))
            DS.ORBIT['cx'], DS.ORBIT['cy'] = float(fx[0]), float(fy[0])
    if os.environ.get('TRANS'):                    # forming / dissolving + speck fades, timed to 146 BPM
        BT = 60.0 / 146
        TL = float(os.environ.get('CLIPLEN', n / 25.0))
        DS.TRANS.update(dict(dir=float(os.environ.get('TRANS_DIR', '1')),
                             in0=0.0, in_spread=1.6 * BT, in_dur=1.7 * BT,
                             out0=TL - 3.2 * BT, out_spread=1.2 * BT, out_dur=1.5 * BT,
                             sp_in0=0.0, sp_in_spread=4 * BT, sp_fade=1.0 * BT,
                             sp_out0=TL - 4.2 * BT, sp_out_spread=3.0 * BT))
    tt_ = (a + i) / 25.0 if TABS else i / 25.0
    if NOTRACK:
        fr_ = DS.render(dep, 1, tt_, None, None, None, ST, lm)
    else:
        fr_ = DS.render(dep, 1, tt_, pts, DS.level_vis(LV, DS.local_scale(pts, SHP)), None, ST, lm)
    if STILL:
        cv2.imwrite(f'{STILL}_{a + i:04d}.png', (np.clip(fr_, 0, 1) * 65535 + 0.5).astype(np.uint16))
    else:
        wr.write(fr_)
    if i + 1 < n and not NOTRACK:
        fl = dis.calc(gray[i], gray[i + 1], None)
        fl = cv2.resize(fl, (W, H)) * (W / 640.0)
        # 1) one coherent motion for the whole flower: homography fitted (RANSAC) to the flow on the flower
        calm = np.clip((cv2.GaussianBlur(dep, (0, 0), 8) - 0.12) / 0.25, 0, 1)
        yy, xx = np.mgrid[8:H:16, 8:W:16]
        m = calm[yy, xx] > 0.5
        p0 = np.stack([xx[m], yy[m]], -1).astype(np.float32)
        p1 = p0 + fl[yy[m], xx[m]]
        Hm = None
        if len(p0) > 40 and os.environ.get('SIMILARITY'):
            A, _ = cv2.estimateAffinePartial2D(p0, p1, method=cv2.RANSAC, ransacReprojThreshold=1.5)
            if A is not None:
                sc = np.hypot(A[0, 0], A[1, 0]); k = np.clip(sc, 0.996, 1.004) / sc
                A[:, :2] *= k
                Hm = np.vstack([A, [0, 0, 1]])
        elif len(p0) > 40:
            Hm, _ = cv2.findHomography(p0, p1, cv2.RANSAC, 1.5)
        if Hm is None:
            Hm = np.eye(3)
        scale = scale * np.sqrt(abs(np.linalg.det(Hm[:2, :2] / Hm[2, 2])))
        P = np.stack([pts[0], pts[1], np.ones_like(pts[0])], 0).astype(np.float64)
        Q = Hm @ P; qx, qy = (Q[0] / Q[2]).astype(np.float32), (Q[1] / Q[2]).astype(np.float32)
        # 2) gentle local correction (opening petals): heavily smoothed residual flow, weighted to the flower
        gx_, gy_ = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
        G = np.stack([gx_, gy_, np.ones_like(gx_)], 0).reshape(3, -1)
        GQ = (Hm @ G); hx = (GQ[0] / GQ[2]).reshape(H, W) - gx_; hy = (GQ[1] / GQ[2]).reshape(H, W) - gy_
        res = (fl - np.stack([hx, hy], -1)) * calm[..., None]
        res = cv2.GaussianBlur(res, (0, 0), 28) / (cv2.GaussianBlur(calm, (0, 0), 28)[..., None] + 1e-3) * calm[..., None]
        res = np.clip(res, -3, 3) * 0.8
        s = DS.sampler(pts)
        pts = (qx + s(res[..., 0]), qy + s(res[..., 1]))
if wr: wr.close()
ls_ = DS.local_scale(pts, SHP); print('final zoom global %.2f, local p50 %.2f p95 %.2f' % (scale[0], np.median(ls_), np.percentile(ls_, 95)))
