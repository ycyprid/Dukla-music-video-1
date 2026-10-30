"""Temporally consistent subject matte for depthsubj shots (blossom): the per-frame matte (depth matte x sky colour
key, exactly as clip_new.py builds it) flickers where depth / colour sit near their thresholds, and depth glitches
make short-lived blobs. Fix: every frame's matte = median of the mattes of its neighbours (+-R frames), each warped
into this frame with optical flow -> anything that exists for fewer than R+1 of the 2R+1 frames is dropped, edges
stop flickering, real motion is kept. Small islands that are not stable are removed too.
usage: python3 blossom_matte.py src.mp4 depthdir n outdir"""
import sys, os
import numpy as np, cv2
import mattes
from common import W, H
EV = lambda k, d: float(os.environ.get(k, d))
src, dd, n, od = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
os.makedirs(od, exist_ok=True)
R = int(EV('MT_R', '3'))
cap = cv2.VideoCapture(src)
frames, gs = [], []
for _ in range(n):
    ok, fr = cap.read()
    f = cv2.resize(fr, (W, H), interpolation=cv2.INTER_AREA); frames.append(f); gs.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
ld = lambda i: cv2.resize(cv2.imread(f'{dd}/d{i:04d}.png', -1).astype(np.float32) / 65535, (W, H))
smp = [ld(i) for i in range(0, n, 5)]
dlo = np.median([np.percentile(x, 2) for x in smp]); dhi = np.median([np.percentile(x, 99.7) for x in smp]); del smp
yq, xq = np.mgrid[0:90, 0:160].astype(np.float32); xq = xq / 160 - 0.5; yq = yq / 90 - 0.5
Bq = np.stack([np.ones_like(xq), xq, yq, xq * xq, xq * yq, yq * yq], -1)
SK = {}
RAW = np.zeros((n, H, W), np.uint8)
for i in range(n):
    d = np.clip((ld(i) - dlo) / (dhi - dlo + 1e-6), 0, 1)
    g8 = cv2.resize(gs[i], (640, 360), interpolation=cv2.INTER_AREA)
    mm, ab = mattes.depth_matte(d, g8, lift=EV('DS_LIFT', '0.07'))
    f32 = frames[i].astype(np.float32) / 255
    sm_ = cv2.resize(f32, (160, 90), interpolation=cv2.INTER_AREA); msk = cv2.resize(mm, (160, 90)) < 0.05
    cq = np.stack([np.linalg.lstsq(Bq[msk], sm_[..., c][msk], rcond=None)[0] for c in range(3)])
    if 'cq' in SK:
        cq = 0.75 * SK['cq'] + 0.25 * cq
    SK['cq'] = cq
    skyc = np.stack([cv2.resize((Bq @ cq[c]).astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC) for c in range(3)], -1)
    dist = np.sqrt(((cv2.GaussianBlur(f32, (0, 0), 0.8) - skyc) ** 2).sum(-1))
    c0, c1 = (float(x_) for x_ in os.environ.get('SKYKEY', '0.05,0.13').split(','))
    al = np.clip((dist - c0) / (c1 - c0), 0, 1); al = al * al * (3 - 2 * al)
    near_ = cv2.GaussianBlur(cv2.dilate(mm, np.ones((21, 21), np.uint8)), (0, 0), 4)
    m0 = np.clip(al * np.clip(near_, 0, 1), 0, 1)
    # in-focus gate: a blossom / branch has a crisp edge nearby; depth haze and soft glow regions don't
    g_ = gs[i].astype(np.float32) / 255; gb_ = cv2.GaussianBlur(g_, (0, 0), 0.7)
    gm_ = np.hypot(cv2.Sobel(gb_, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(gb_, cv2.CV_32F, 0, 1, ksize=3)) / 4
    sharp = cv2.morphologyEx((gm_ > EV('MT_EDGE', '0.05')).astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    dist_ = cv2.distanceTransform(1 - sharp, cv2.DIST_L2, 5)
    gate = np.clip((EV('MT_GD', '22') - dist_) / 8, 0, 1)
    h = np.clip((m0 * gate - 0.35) / 0.3, 0, 1); h = h * h * (3 - 2 * h)                # firm edge: no half-present haze
    # the gate must not punch holes into big petals (far from any crisp edge): enclosed gaps that were subject
    # in the raw matte are restored
    hb = (h > 0.5).astype(np.uint8)
    inv = 1 - hb
    nc_, lab_, st_, _ = cv2.connectedComponentsWithStats(inv, 4)
    border = np.unique(np.concatenate([lab_[0], lab_[-1], lab_[:, 0], lab_[:, -1]]))
    HA = EV('MT_HOLE', '4000')
    hw_ = np.zeros(nc_, np.float32)
    for k in range(1, nc_):
        if k not in border:
            hw_[k] = np.clip((HA * 1.5 - st_[k, cv2.CC_STAT_AREA]) / HA, 0, 1)     # fills fade out with hole size
    hwm = hw_[lab_]
    h = np.maximum(h, hwm * np.clip((m0 - 0.35) / 0.3, 0, 1))
    OPK = int(EV('MT_OPEN', '0'))
    if OPK > 0:                                   # structures thinner than the dot spacing alias into flicker: drop them
        h = cv2.morphologyEx(h.astype(np.float32), cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (OPK, OPK)))
    RAW[i] = (h * 255 + 0.5).astype(np.uint8)
print('raw mattes done', flush=True)
# ---- temporal filter along the motion: bidirectional recursive average (forward + backward pass), each step warped
#      with optical flow and trusted only where forward/backward flow agree. A blossom that the per-frame matte
#      switches on/off for a few frames becomes a gentle fade (or vanishes), never a pop.
dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
SW_, SH_ = 640, 360
small = [cv2.resize(g, (SW_, SH_), interpolation=cv2.INTER_AREA) for g in gs]
gxs, gys = np.meshgrid(np.arange(SW_, dtype=np.float32), np.arange(SH_, dtype=np.float32))
gx, gy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
A = EV('MT_A', '0.82')
def flowpair(i, j):
    f = dis.calc(small[i], small[j], None); b = dis.calc(small[j], small[i], None)
    bx = cv2.remap(b[..., 0], gxs + f[..., 0], gys + f[..., 1], cv2.INTER_LINEAR); by = cv2.remap(b[..., 1], gxs + f[..., 0], gys + f[..., 1], cv2.INTER_LINEAR)
    err = np.hypot(f[..., 0] + bx, f[..., 1] + by)
    trust = np.clip(1.5 - err, 0, 1)                                    # >1.5 px disagreement: occlusion / unreliable
    return cv2.resize(f, (W, H)) * (W / SW_), cv2.resize(trust, (W, H))
def run(order):
    out = {}; prev = None; pi = None
    for i in order:
        r = RAW[i].astype(np.float32) / 255
        if prev is None:
            cur = r
        else:
            fl, tr = flowpair(i, pi)
            w = cv2.remap(prev, gx + fl[..., 0], gy + fl[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            a = A * tr
            cur = a * w + (1 - a) * r
        out[i] = cur; prev = cur; pi = i
    return out
Fw = run(range(n)); print('forward done', flush=True)
Bw = run(range(n - 1, -1, -1)); print('backward done', flush=True)
flick = []
for i in range(n):
    r = RAW[i].astype(np.float32) / 255
    S = (Fw[i] + Bw[i] - (1 - A) * r) / (1 + A)                          # both directions (raw counted once)
    m = np.clip((S - 0.30) / 0.40, 0, 1); m = m * m * (3 - 2 * m)          # soft threshold: changes fade over frames
    m = cv2.GaussianBlur(m, (0, 0), 0.8)
    flick.append(float(np.abs(m - r).mean()))
    cv2.imwrite(f'{od}/m{i:04d}.png', (np.clip(m, 0, 1) * 255 + 0.5).astype(np.uint8))
np.save(f'{od}/change.npy', np.array(flick)); print('done', flush=True)
