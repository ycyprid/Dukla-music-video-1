"""Motion tests for the silhouette / keyed looks (same maths as stills_new.py 'backlit' and 'keyed', per frame).
No per-frame depth (d = constant): these looks take their form from the footage's light, so a depth pass would
only add flicker. Levels (sky model, reference light) are smoothed over time so nothing pumps.
usage: python3 motion_sil.py src.mp4 start n mode out.mp4      mode: backlit | keyed
env: CROP=x,y,w (16:9 window, source px); TRACK=1 makes that window follow the subjects (smoothed)"""
import sys, os
import numpy as np, cv2
from scipy.ndimage import gaussian_filter1d
import depth_sheet as DS
import common
from common import W, H, Writer
common.FPS = 25
src, a, n, mode, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5]
cap = cv2.VideoCapture(src)
for _ in range(a):
    cap.grab()
F = []
for _ in range(n):
    ok, f = cap.read()
    F.append(f)
EV = lambda k, d: float(os.environ.get(k, d))
d = 0.5


def quad(sm, msk):
    h_, w_ = sm.shape[:2]
    yy, xx = np.mgrid[0:h_, 0:w_].astype(np.float32); xx = xx / w_ - 0.5; yy = yy / h_ - 0.5
    Bq = np.stack([np.ones_like(xx), xx, yy, xx * xx, xx * yy, yy * yy], -1)
    return Bq, Bq[msk]


# ---- crop window per frame (optionally tracking the subjects) ----
if os.environ.get('CROP'):
    cx0, cy0, cw = (int(float(x_)) for x_ in os.environ['CROP'].split(',')); ch = int(round(cw * 9 / 16))
    ox = np.full(n, float(cx0)); oy = np.full(n, float(cy0))
    if os.environ.get('TRACK'):
        cents = []
        for f in F:                                   # subjects = far from a robust sky-colour fit (low res)
            sm = cv2.resize(f, (240, 135), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
            msk = np.ones((135, 240), bool)
            for _ in range(3):
                Bq, Bm = quad(sm, msk)
                fit = np.stack([Bq @ np.linalg.lstsq(Bm, sm[..., c][msk], rcond=None)[0] for c in range(3)], -1)
                r_ = np.sqrt(((sm - fit) ** 2).sum(-1)); msk = r_ < max(np.percentile(r_, 90), 0.01)
            s_ = 1920 / 240
            yy_, xx_ = np.nonzero(r_ > 0.09)
            x_ = xx_ * s_; y_ = yy_ * s_
            # only what is (roughly) inside the styleframe window, with a margin
            k_ = (x_ > cx0 - 60) & (x_ < cx0 + cw + 60) & (y_ > cy0 - 60) & (y_ < cy0 + ch + 60)
            cents.append((x_[k_].mean(), y_[k_].mean()) if k_.sum() > 5 else (np.nan, np.nan))
        cents = np.array(cents); ok_ = ~np.isnan(cents[:, 0])
        for q in range(2):
            cents[:, q] = np.interp(np.arange(n), np.nonzero(ok_)[0], cents[ok_, q])
        cs = gaussian_filter1d(cents, 10, axis=0, mode='nearest')
        ref = cs[int(os.environ.get('TRACK_REF', n // 2))]
        ox = cx0 + (cs[:, 0] - ref[0]) * EV('TRACK_K', '0.85'); oy = cy0 + (cs[:, 1] - ref[1]) * EV('TRACK_K', '0.85')
        ox = np.clip(ox, 0, 1920 - cw); oy = np.clip(oy, 0, 1080 - ch)
        print('crop travel px', round(np.ptp(ox)), round(np.ptp(oy)))


def frame_in(i):
    f = F[i]
    if os.environ.get('CROP'):
        # sub-pixel crop (smooth tracking): affine warp of the window to the design size
        sx = cw / W
        M = np.array([[1 / sx, 0, -ox[i] / sx], [0, 1 / sx, -oy[i] / sx]], np.float32)
        return cv2.warpAffine(f, M, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT101)
    return cv2.resize(f, (W, H), interpolation=cv2.INTER_AREA)


ST = {}
def smooth(key, val, k=0.15):
    ST[key] = val if key not in ST else ST[key] + (val - ST[key]) * k
    return ST[key]


wr = Writer(out, None, [])
for i in range(n):
    frw = frame_in(i)
    g = cv2.cvtColor(frw, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
    fab = None
    if mode == 'backlit':
        gb = cv2.GaussianBlur(g, (0, 0), 1.0)
        sm = cv2.resize(gb, (192, 108), interpolation=cv2.INTER_AREA)
        det = np.abs(sm - cv2.GaussianBlur(sm, (0, 0), 2))
        cand = (sm > np.percentile(sm, EV('BL_SKYP', '45'))) & (det < 0.04)
        yy, xx = np.mgrid[0:108, 0:192].astype(np.float32); xx /= 192; yy /= 108
        Bs = np.stack([np.ones_like(xx), xx, yy, xx * xx, xx * yy, yy * yy], -1)
        coef = np.linalg.lstsq(Bs[cand], sm[cand], rcond=None)[0]
        skys = smooth('sky', np.clip(Bs @ coef, 0.6 * np.median(sm[cand]), 1.0).astype(np.float32))
        sky = cv2.resize(skys, (W, H), interpolation=cv2.INTER_CUBIC)
        m = np.clip((sky * EV('BL_T', '0.55') - gb) / (0.10 * sky + 0.01), 0, 1)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        m = cv2.GaussianBlur(m, (0, 0), 0.8)
        out_ = cv2.GaussianBlur(1 - m, (0, 0), EV('BL_RIMW', '2.2'))
        rim = np.clip(m * out_ * 2.2, 0, 1) ** 0.8
        rim = rim * np.clip(sky / (np.percentile(sky, 90) + 1e-6), 0.25, 1) ** 0.6
        ref = smooth('ref', float(np.percentile(gb[m < 0.2], 99.5)) if (m < 0.2).sum() > 100 else 1.0, 0.1)
        sl = np.clip(gb / ref, 0, 1.1) ** EV('BL_G', '1.5')
        fg = (m > 0.5).astype(np.uint8); fg[int(H * EV('BL_GROUND', '1.0')):] = 0
        dt = cv2.distanceTransform(1 - fg, cv2.DIST_L2, 5) / H
        aura = np.exp(-dt / EV('BL_AURA', '0.22'))
        AF = EV('BL_FLOOR', '0.22')
        sl = sl * (AF + (1 - AF) * aura)
        lum = np.clip((1 - m) * sl * EV('BL_SKY', '0.85') + EV('BL_RIM', '1.2') * rim, 0, 1.2)
        dep = np.clip((1 - m) * (0.25 + 0.30 * sl) + rim * (0.55 + 0.45 * d), 0, 1)
    else:                                     # keyed (birds)
        f32 = frw.astype(np.float32) / 255
        sm_ = cv2.resize(f32, (160, 90), interpolation=cv2.INTER_AREA)
        msk = np.ones((90, 160), bool)
        for _ in range(3):
            Bq, Bm = quad(sm_, msk)
            cf = [np.linalg.lstsq(Bm, sm_[..., c][msk], rcond=None)[0] for c in range(3)]
            fit = np.stack([Bq @ c_ for c_ in cf], -1)
            r_ = np.sqrt(((sm_ - fit) ** 2).sum(-1)); msk = r_ < max(np.percentile(r_, 90), 0.01)
        fit = smooth('skyc', fit)
        skyc = np.stack([cv2.resize(fit[..., c].astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC) for c in range(3)], -1)
        dist = np.sqrt(((cv2.GaussianBlur(f32, (0, 0), 0.7) - skyc) ** 2).sum(-1))
        c0, c1 = (float(x_) for x_ in os.environ.get('SKYKEY', '0.05,0.14').split(','))
        al = np.clip((dist - c0) / (c1 - c0), 0, 1); al = cv2.GaussianBlur(al * al * (3 - 2 * al), (0, 0), 0.7)
        sel = al > 0.5
        ref = smooth('ref', float(np.percentile(g[sel], 95)) if sel.sum() > 30 else ST.get('ref', 1.0), 0.1)
        gl_ = cv2.GaussianBlur(np.clip(g / ref, 0, 1), (0, 0), 1.2)
        dep = al * (EV('KEY_D0', '0.58') + 0.30 * gl_ + 0.12 * d)
        lum = None
        if os.environ.get('FABRIC'):
            gg_ = cv2.GaussianBlur(g, (0, 0), 0.6)
            hp_ = np.abs(gg_ - cv2.GaussianBlur(gg_, (0, 0), 2.2))
            fab = (np.clip(hp_ / 0.05, 0, 1) * np.clip(dep / 0.5, 0, 1)).astype(np.float32)
    img = DS.render(dep.astype(np.float32), 1, i / 25.0, None, None, None, None,
                    None if lum is None else lum.astype(np.float32), fab)
    wr.write(img)
wr.close()
print('done', out)
