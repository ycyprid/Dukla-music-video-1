"""Gulls, full shot: keyed look (as the style frame / 3 s test) + per-bird dot grids (depth_sheet_objects) so the
pigeon-style flight transitions work: forming streams in along each bird's own flight path, tail first; dispersal
streams on ahead of each bird along its flight, head first. Roof / vignette corners (touching the source frame
edge) are removed. usage: python3 gulls_full.py src.mp4 outdir [f0 f1]"""
import sys, os
import numpy as np, cv2
from scipy.ndimage import gaussian_filter1d
import depth_sheet as DS
import depth_sheet_objects as O
from common import W, H

EV = lambda k, d: float(os.environ.get(k, d))
src, od = sys.argv[1], sys.argv[2]
os.makedirs(od, exist_ok=True)
cap = cv2.VideoCapture(src)
F = []
while True:
    ok, f = cap.read()
    if not ok:
        break
    F.append(f)
N = len(F); TL = N / 25.0
f0, f1 = (int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) > 4 else (0, N)


def quad(sm, msk):
    h_, w_ = sm.shape[:2]
    yy, xx = np.mgrid[0:h_, 0:w_].astype(np.float32); xx = xx / w_ - 0.5; yy = yy / h_ - 0.5
    Bq = np.stack([np.ones_like(xx), xx, yy, xx * xx, xx * yy, yy * yy], -1)
    return Bq, Bq[msk]


def robust_resid(sm):
    msk = np.ones(sm.shape[:2], bool)
    for _ in range(3):
        Bq, Bm = quad(sm, msk)
        fit = np.stack([Bq @ np.linalg.lstsq(Bm, sm[..., c][msk], rcond=None)[0] for c in range(3)], -1)
        r_ = np.sqrt(((sm - fit) ** 2).sum(-1)); msk = r_ < max(np.percentile(r_, 90), 0.01)
    return r_, fit


# ---- per frame (low res, source coords): what touches the frame edge (roof, vignette corners) + flock centre ----
KILL = []; CEN = []
s_ = 1920 / 480
for f in F:
    sm = cv2.resize(f, (480, 270), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    r_, _ = robust_resid(sm)
    bw = (r_ > 0.06).astype(np.uint8)
    n_, lab, st, cen = cv2.connectedComponentsWithStats(bw, 8)
    kill = np.zeros((270, 480), np.uint8); pts = []
    for k in range(1, n_):
        x0, y0, w0, h0, a0 = st[k]
        edge = x0 <= 1 or y0 <= 1 or x0 + w0 >= 479 or y0 + h0 >= 269
        if edge:
            kill[lab == k] = 1
        elif a0 >= 3 and r_[lab == k].max() > 0.09:
            pts.append(cen[k] * s_)
    KILL.append(cv2.dilate(kill, np.ones((5, 5), np.uint8)))
    pts = np.array(pts) if pts else np.zeros((0, 2))
    if len(pts) >= 4:                                     # the lone 4th gull heading off right: not part of the framing
        md = np.median(pts, 0); dd = np.hypot(*(pts - md).T); pts = pts[np.argsort(dd)[:3]]
    CEN.append(((pts.min(0) + pts.max(0)) / 2) if len(pts) else (np.nan, np.nan))
CEN = np.array(CEN, float); ok_ = ~np.isnan(CEN[:, 0])
for q in range(2):
    CEN[:, q] = np.interp(np.arange(N), np.nonzero(ok_)[0], CEN[ok_, q])
CS = gaussian_filter1d(CEN, EV('TRACK_SIG', '14'), axis=0, mode='nearest')
_x = np.clip(np.arange(N) / EV('ZOOM_F', '130'), 0, 1); _x = _x * _x * (3 - 2 * _x)
CWa = EV('CROPW0', '720') + (EV('CROPW', '1060') - EV('CROPW0', '720')) * _x     # tighter while the gulls are far
CHa = CWa * 9 / 16
OX = np.clip(CS[:, 0] - CWa / 2, 0, 1920 - CWa); OY = np.clip(CS[:, 1] - CHa / 2 + EV('CROP_DY', '0'), 0, 1080 - CHa)
FV = np.gradient(gaussian_filter1d(CEN, 25, axis=0, mode='nearest'), axis=0)        # flock flight (source px / frame)
print('crop travel', np.ptp(OX).round(), np.ptp(OY).round(), flush=True)


def window(i):
    sx = CWa[i] / W
    M = np.array([[1 / sx, 0, -OX[i] / sx], [0, 1 / sx, -OY[i] / sx]], np.float32)
    f = cv2.warpAffine(F[i], M, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT101)
    k = cv2.resize(KILL[i].astype(np.float32), (1920, 1080), interpolation=cv2.INTER_LINEAR)
    k = cv2.warpAffine(k, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=0)
    return f, np.clip(k * 2, 0, 1)


ST_ = {}
def smooth(key, val, k=0.12):
    ST_[key] = val if key not in ST_ else ST_[key] + (val - ST_[key]) * k
    return ST_[key]


def keyed(frw, kill):
    g = cv2.cvtColor(frw, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
    f32 = frw.astype(np.float32) / 255
    sm_ = cv2.resize(f32, (160, 90), interpolation=cv2.INTER_AREA)
    _, fit = robust_resid(sm_)
    fit = smooth('skyc', fit)
    skyc = np.stack([cv2.resize(fit[..., c].astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC) for c in range(3)], -1)
    dist = np.sqrt(((cv2.GaussianBlur(f32, (0, 0), 0.7) - skyc) ** 2).sum(-1))
    c0, c1 = (float(x_) for x_ in os.environ.get('SKYKEY', '0.05,0.14').split(','))
    al = np.clip((dist - c0) / (c1 - c0), 0, 1); al = cv2.GaussianBlur(al * al * (3 - 2 * al), (0, 0), 0.7)
    al = al * (1 - kill)
    sel = al > 0.5
    ref = smooth('ref', float(np.percentile(g[sel], 95)) if sel.sum() > 30 else ST_.get('ref', 1.0), 0.1)
    gl_ = cv2.GaussianBlur(np.clip(g / ref, 0, 1), (0, 0), 1.2)
    dep = al * (EV('KEY_D0', '0.58') + 0.30 * gl_ + 0.06)
    gg_ = cv2.GaussianBlur(g, (0, 0), 0.6)
    hp_ = np.abs(gg_ - cv2.GaussianBlur(gg_, (0, 0), 2.2))
    fab = (np.clip(hp_ / 0.05, 0, 1) * np.clip(dep / 0.5, 0, 1)).astype(np.float32)
    return dep.astype(np.float32), fab, g


BT = 60.0 / 146
DS.TRANS.update(dict(mode='flow', fdir=(0.0, -1.0), in0=0.04, in_spread=1.6 * BT, in_dur=1.9 * BT,
                     bird_out=True, out_len=EV('OUT_LEN', '300'),
                     out_spread=1.6 * BT, out_dur=1.9 * BT, out0=TL - (1.6 + 1.9 * 1.2) * BT - 0.12,
                     sw_in_dur=2.0, sw_out0=TL - 2.6, sw_out_dur=2.3))
P = DS.PITCH_PX
sky_x, sky_y = DS.lattice()
sky_id = (9_000_000_000 + np.arange(sky_x.size)).astype(np.int64)
sxi = np.clip(np.round(sky_x).astype(int), 0, W - 1); syi = np.clip(np.round(sky_y).astype(int), 0, H - 1)
objs = []
ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * O.DIL + 1, 2 * O.DIL + 1))
dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
ST = {}
prev_g = None; prev_dep = None
w0 = max(0, f0 - 8)
for i in range(w0, f1):
    frw, kill = window(i)
    dep, fab, g = keyed(frw, kill)
    g8 = cv2.resize((g * 255).astype(np.uint8), (640, 360), interpolation=cv2.INTER_AREA)
    # move objects with the flow from the previous frame (before matching)
    if prev_g is not None:
        fl = cv2.resize(dis.calc(prev_g, g8, None), (W, H)) * (W / 640.0)
        for o in objs:
            if o.mask is not None:
                A = O.fit_motion(fl, o.mask)
                if A is not None:
                    o.apply(A)
    m = (cv2.GaussianBlur(dep, (0, 0), 1.5) > O.THR).astype(np.uint8)
    nc, lab, st, cen = cv2.connectedComponentsWithStats(m, 8)
    comps = [k for k in range(1, nc) if st[k, cv2.CC_STAT_AREA] >= 40]
    claim = {k: [] for k in comps}; pairs = []
    for oi, o in enumerate(objs):
        for k in comps:
            dd_ = np.hypot(*(cen[k] - o.c)); tol = max(70.0, 0.9 * np.sqrt(st[k, cv2.CC_STAT_AREA]) + np.hypot(*o.vel))
            if dd_ < tol:
                pairs.append((dd_, oi, k))
    pairs.sort(); used_o, used_k = set(), set()
    for dd_, oi, k in pairs:
        if oi in used_o or k in used_k:
            continue
        used_o.add(oi); used_k.add(k); claim[k].append(objs[oi])
    for oi, o in enumerate(objs):
        if oi not in used_o:
            o.miss += 1; o.mask = None
        else:
            o.miss = 0
    # flock (crop) motion this frame, in design px: added back so each bird's direction is its real flight
    for k in comps:
        if not claim[k]:
            o = O.Obj(*cen[k]); objs.append(o); claim[k].append(o)
        else:
            o = claim[k][0]
            o.vel = 0.5 * o.vel + 0.5 * (cen[k] - o.c)
            o.c = cen[k].astype(np.float64).copy()
    objs = [o for o in objs if o.miss <= 4]
    X, Y, ID, FX, FY = [sky_x[:0]], [sky_y[:0]], [sky_id[:0]], [np.zeros(0)], [np.zeros(0)]
    anyobj = np.zeros((H, W), bool)
    for k, os_ in claim.items():
        o = os_[0]; r = lab == k
        o.mask = r
        rd = cv2.dilate(r.astype(np.uint8), ker).astype(bool); anyobj |= rd
        x_, y_, id_ = O.obj_dots(o, rd)
        X.append(x_); Y.append(y_); ID.append(id_)
        fv = FV[i] if np.hypot(*FV[i]) > 1e-3 else np.array([0.0, -1.0])
        if os.environ.get("DBGFV") and i % 10 == 0: print("FV", i, FV[i].round(2), flush=True)
        FX.append(np.full(x_.size, fv[0])); FY.append(np.full(x_.size, fv[1]))
    keep = ~anyobj[syi, sxi]
    X.append(sky_x[keep]); Y.append(sky_y[keep]); ID.append(sky_id[keep])
    FX.append(np.zeros(keep.sum())); FY.append(-np.ones(keep.sum()))
    X, Y, ID = np.concatenate(X), np.concatenate(Y), np.concatenate(ID)
    DS.TRANS['fdir_dot'] = (np.concatenate(FX), np.concatenate(FY))
    t = i / 25.0
    img = DS.render(dep, 1, t, (X, Y), None, ID, ST, None, fab)
    if i >= f0:
        cv2.imwrite(f'{od}/gulls_{i:04d}.png', (np.clip(img, 0, 1) * 65535 + 0.5).astype(np.uint16))
        print(i, len(objs), flush=True)
    prev_g = g8
