"""Depth dot-sheet with per-object tracking (several independently moving subjects, e.g. birds).
Each subject (connected region of near depth) carries its own dot grid in its own local frame; the frame
moves/rotates/scales with that subject (similarity fitted to the optical flow on it), so its dots stay
attached and its density stays even while wings change shape (the subject's mask just clips its grid).
Empty background keeps a fixed faint grid. Glints are tied to each dot's identity (no flicker).
usage: python3 depth_sheet_objects.py depthdir start n out.mp4 src.mp4 src_offset"""
import sys, glob
import numpy as np, cv2
import os
import depth_sheet as DS
import mattes
from common import Writer, W, H
import common
common.FPS = 25

P = DS.PITCH_PX
MIN_AREA = 120            # px at 1280x720
THR = 0.30                # normalised depth above which a pixel belongs to a subject
DIL = 7                   # mask dilation (dots reach the soft edge)


class Obj:
    nid = 0

    def __init__(self, cx, cy):
        Obj.nid += 1
        self.id = Obj.nid
        self.T = np.array([[1, 0, cx], [0, 1, cy]], np.float64)   # local -> screen (similarity)
        self.c = np.array([cx, cy], np.float64); self.miss = 0; self.mask = None; self.vel = np.zeros(2)

    def apply(self, A):                                               # compose a screen-space motion
        A3 = np.vstack([A, [0, 0, 1]]); T3 = np.vstack([self.T, [0, 0, 1]])
        T = (A3 @ T3)[:2]
        sc = np.sqrt(abs(np.linalg.det(T[:, :2])))
        T[:, :2] *= np.clip(sc, 0.8, 1.25) / sc                 # grid spacing stays near the house pitch
        self.T = T
        self.c = A @ np.array([self.c[0], self.c[1], 1.0])


def fit_motion(fl, mask):
    ys, xs = np.nonzero(mask[::2, ::2]); ys *= 2; xs *= 2
    if len(xs) < 12:
        return None
    if len(xs) > 1500:
        k = np.random.default_rng(0).choice(len(xs), 1500, replace=False); xs, ys = xs[k], ys[k]
    p0 = np.stack([xs, ys], -1).astype(np.float32)
    p1 = p0 + fl[ys, xs]
    A, _ = cv2.estimateAffinePartial2D(p0, p1, method=cv2.RANSAC, ransacReprojThreshold=1.5)
    if A is not None:                                   # flapping changes the area, not the bird's size: tiny scale steps only
        sc = np.hypot(A[0, 0], A[1, 0]); A[:, :2] *= np.clip(sc, 0.996, 1.004) / sc
    return A


def obj_dots(o, region):
    """local integer grid of object o, kept where it lands inside its screen region"""
    ys, xs = np.nonzero(region)
    if len(xs) == 0:
        return np.zeros(0, np.float32), np.zeros(0, np.float32), np.zeros(0, np.int64)
    Ti = cv2.invertAffineTransform(o.T)
    corners = np.array([[xs.min(), ys.min(), 1], [xs.max(), ys.min(), 1], [xs.min(), ys.max(), 1], [xs.max(), ys.max(), 1]], np.float64)
    lc = corners @ Ti.T
    i0, i1 = int(np.floor(lc[:, 0].min() / P)) - 1, int(np.ceil(lc[:, 0].max() / P)) + 1
    j0, j1 = int(np.floor(lc[:, 1].min() / P)) - 1, int(np.ceil(lc[:, 1].max() / P)) + 1
    I, J = np.meshgrid(np.arange(i0, i1 + 1), np.arange(j0, j1 + 1))
    I = I.ravel(); J = J.ravel()
    q = np.stack([I * P, J * P, np.ones(I.size)], 0)
    sx, sy = o.T @ q
    xi = np.clip(np.round(sx).astype(int), 0, W - 1); yi = np.clip(np.round(sy).astype(int), 0, H - 1)
    ok = region[yi, xi] & (sx >= -2) & (sx < W + 2) & (sy >= -2) & (sy < H + 2)
    ids = (o.id * 1_000_000 + (I[ok] + 500) * 1000 + (J[ok] + 500)).astype(np.int64)
    return sx[ok].astype(np.float32), sy[ok].astype(np.float32), ids


def main():
    dd, a, n, out, src, off = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5], int(sys.argv[6])
    fs = sorted(glob.glob(f'{dd}/d*.png'))[a:a + n]
    allf = sorted(glob.glob(f'{dd}/d*.png'))
    smp = [cv2.resize(cv2.imread(f, -1).astype(np.float32) / 65535, (320, 180)) for f in allf[::6]]
    lohi = (np.median([np.percentile(x, 2) for x in smp]), np.percentile(np.stack(smp), 99.8))
    cap = cv2.VideoCapture(src)
    for _ in range(off + a):
        cap.grab()
    gray = []; grayf = []
    for _ in range(n):
        ok, fr = cap.read()
        gray.append(cv2.cvtColor(cv2.resize(fr, (640, 360), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY))
        grayf.append(cv2.cvtColor(cv2.resize(fr, (W, H), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY))
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    STILL = os.environ.get('STILL'); TABS = os.environ.get('T0ABS')
    if os.environ.get('TRANS'):
        # flight direction of the flock over the opening (birds' own motion), for the forming streams
        fv = []
        for j in range(min(n - 1, 30)):
            fl_ = dis.calc(gray[j], gray[j + 1], None); mk = mattes.sky_matte(gray[j], 640, 360) > 0.5
            if mk.sum() > 50:
                fv.append(fl_[mk].mean(0))
        fdir = np.mean(fv, 0) if fv else np.array([0.0, -1.0])
        fdir = fdir / (np.linalg.norm(fdir) + 1e-9)
        BT = 60.0 / 146; TL = float(os.environ.get('CLIPLEN', n / 25.0))
        DS.TRANS.update(dict(mode='flow', fdir=(float(fdir[0]), float(fdir[1])),
                             in0=0.0, in_spread=1.6 * BT, in_dur=1.9 * BT,
                             bg_out0=TL - 2.4, bg_out_dur=2.0,
                             sw_in_dur=2.0, sw_out0=TL - 2.6, sw_out_dur=2.3))
        print('flight direction %.3f, %.3f' % tuple(fdir), flush=True)
    sky_x, sky_y = DS.lattice()
    sky_id = (9_000_000_000 + np.arange(sky_x.size)).astype(np.int64)
    sxi = np.clip(np.round(sky_x).astype(int), 0, W - 1); syi = np.clip(np.round(sky_y).astype(int), 0, H - 1)
    objs = []
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * DIL + 1, 2 * DIL + 1))
    wr = None if STILL else Writer(out, None, [])
    ST = {}
    nobj = []
    for i, f in enumerate(fs):
        dep = DS.load_depth(f, lohi)
        if os.environ.get('SKYMATTE'):                                  # exact bird outlines from the footage
            sm_ = mattes.sky_matte(gray[i], W, H)
            if os.environ.get('DEPTHMATTE'):                            # pale underwings (as bright as the sky): depth says bird
                dm_ = np.clip((cv2.GaussianBlur(dep, (0, 0), 1.0) - 0.30) / 0.15, 0, 1)
                key_ = (sm_ > 0.5).astype(np.uint8)
                closed = cv2.morphologyEx(key_, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41)))
                ff = closed.copy(); msk = np.zeros((H + 2, W + 2), np.uint8)
                cv2.floodFill(ff, msk, (0, 0), 1)                                   # background reachable from outside
                holes = (ff == 0).astype(np.uint8)                                  # enclosed by the bird's outline
                near_ = np.clip(closed + holes, 0, 1).astype(np.float32)            # inside the keyed outline only
                near_ = cv2.erode(near_, np.ones((3, 3), np.uint8))
                sm_ = np.maximum(sm_, dm_ * near_)
            dep = dep * sm_
        m = (cv2.GaussianBlur(dep, (0, 0), 1.5) > THR).astype(np.uint8)
        nc, lab, st, cen = cv2.connectedComponentsWithStats(m, 8)
        comps = [k for k in range(1, nc) if st[k, cv2.CC_STAT_AREA] >= MIN_AREA]
        # assign components to tracked objects: greedy nearest match between each object's predicted centre and the
        # component centroids, tolerance scaled by the bird's size (+ its speed); matched objects re-anchor to the
        # component centroid every frame, so the prediction never drifts off a fast, flapping bird
        claim = {k: [] for k in comps}
        pairs = []
        for oi, o in enumerate(objs):
            for k in comps:
                dd_ = np.hypot(*(cen[k] - o.c))
                tol = max(70.0, 0.9 * np.sqrt(st[k, cv2.CC_STAT_AREA]) + np.hypot(*o.vel))
                if dd_ < tol:
                    pairs.append((dd_, oi, k))
        pairs.sort()
        used_o, used_k = set(), set()
        for dd_, oi, k in pairs:
            if oi in used_o or k in used_k:
                continue
            used_o.add(oi); used_k.add(k); claim[k].append(objs[oi])
        # a second object may share a component (overlapping birds) if it is clearly inside it
        for oi, o in enumerate(objs):
            if oi in used_o:
                continue
            cx, cy = int(np.clip(o.c[0], 0, W - 1)), int(np.clip(o.c[1], 0, H - 1))
            k = lab[cy, cx]
            if k in claim and claim[k]:
                claim[k].append(o); used_o.add(oi)
        for oi, o in enumerate(objs):
            if oi in used_o:
                o.miss = 0
            else:
                o.miss += 1; o.mask = None
        for k in comps:
            if not claim[k]:
                o = Obj(*cen[k]); objs.append(o); claim[k].append(o)
            elif len(claim[k]) == 1:
                o = claim[k][0]
                o.vel = 0.5 * o.vel + 0.5 * (cen[k] - o.c)              # smoothed screen velocity (for the tolerance)
                o.c = cen[k].astype(np.float64).copy()                   # re-anchor: no drift
        objs = [o for o in objs if o.miss <= 4]
        # regions: a component shared by several objects is split by nearest centre
        X, Y, ID = [sky_x[:0]], [sky_y[:0]], [sky_id[:0]]
        anyobj = np.zeros((H, W), bool)
        for k, os_ in claim.items():
            comp = lab == k
            if len(os_) == 1:
                parts = [(os_[0], comp)]
            else:
                yy, xx = np.nonzero(comp)
                dists = np.stack([np.hypot(xx - o.c[0], yy - o.c[1]) for o in os_], 0)
                who = np.argmin(dists, 0)
                parts = []
                for j, o in enumerate(os_):
                    r = np.zeros((H, W), bool); r[yy[who == j], xx[who == j]] = True; parts.append((o, r))
            for pi_, (o, r) in enumerate(parts):
                o.mask = r
                rd = cv2.dilate(r.astype(np.uint8), ker).astype(bool)
                if len(parts) > 1:                                       # don't spill into a neighbour's half (no seam)
                    for pj_, (_, r2) in enumerate(parts):
                        if pj_ != pi_:
                            rd &= ~r2
                anyobj |= rd
                x_, y_, id_ = obj_dots(o, rd)
                X.append(x_); Y.append(y_); ID.append(id_)
        keep = ~anyobj[syi, sxi]                                         # background grid where no subject is
        X.append(sky_x[keep]); Y.append(sky_y[keep]); ID.append(sky_id[keep])
        X, Y, ID = np.concatenate(X), np.concatenate(Y), np.concatenate(ID)
        fab = None
        if os.environ.get('FABRIC'):
            # feather detail: fine dark/light structure inside the birds (wing bars, feather edges, tail tips)
            gi_ = 1 - grayf[i].astype(np.float32) / 255
            hp = np.abs(gi_ - cv2.GaussianBlur(gi_, (0, 0), 2.0))
            bm = cv2.erode((mattes.sky_matte(gray[i], W, H) > 0.5).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(np.float32)
            fab = np.clip(hp / float(os.environ.get('FAB_HP', '0.06')), 0, 1) ** 1.2 * cv2.GaussianBlur(bm, (0, 0), 1.0)
        tt_ = (a + i) / 25.0 if TABS else i / 25.0
        RNG_ = [int(x_) for x_ in os.environ['RANGE'].split(',')] if os.environ.get('RANGE') else None
        if RNG_ and not (RNG_[0] - 6 <= a + i < RNG_[1]):     # chunked export: tracking runs on, rendering only in range
            fr_ = None
        else:
            fr_ = DS.render(dep, 1, tt_, (X, Y), None, ID, ST, None, fab)
        if fr_ is None or (RNG_ and a + i < RNG_[0]):
            pass
        elif STILL:
            cv2.imwrite(f'{STILL}_{a + i:04d}.png', (np.clip(fr_, 0, 1) * 65535 + 0.5).astype(np.uint16))
        else:
            wr.write(fr_)
        nobj.append(sum(o.mask is not None for o in objs))
        if os.environ.get('OBJLOG'):
            print('F%03d ' % (a + i) + ' '.join('%d@(%d,%d)%s' % (o.id, o.c[0], o.c[1], '' if o.mask is not None else 'x') for o in objs), flush=True)
        # move every object with the flow measured on it
        if i + 1 < n:
            fl = cv2.resize(dis.calc(gray[i], gray[i + 1], None), (W, H)) * (W / 640.0)
            for o in objs:
                if o.mask is not None:
                    A = fit_motion(fl, o.mask)
                    if A is not None:
                        o.apply(A)
    if wr:
        wr.close()
    print('objects per frame: min %d max %d, total ids %d' % (min(nobj), max(nobj), Obj.nid))


if __name__ == '__main__':
    main()
