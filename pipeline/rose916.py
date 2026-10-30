"""9:16 side-screen loop: the red rose (Red_rose_SF.obj + colour / normal / occlusion maps) turning on a turntable in
the dot-sheet look. Once per loop a soft wave runs up the rose (stem foot -> bloom): ahead of it the surface shivers
(normal ripple), then peels off in staggered layers into a slow fountain -- up in a spiral close around the stem,
over the bloom, down around the outside, back in at the foot -- and the same particles rebuild the rose from the
bottom up. Everything (rotation, wave, fountain, glints, dust) repeats exactly at T -> seamless loop.
Framing: fitted to the largest outline over the whole turn incl. the fountain envelope -> nothing ever touches an edge.
usage: DESIGN=720x1280 python3 rose916.py obj colour.jpg normal.jpg ao.jpg outdir f0 f1 [t_list]"""
import sys, os
import numpy as np, cv2
import common
import depth_sheet as DS
import sheet_loop2 as L
from common import W, H

EV = lambda k, d: float(os.environ.get(k, d))
objp, colp, nrmp, aop, od = sys.argv[1:6]
f0, f1 = int(sys.argv[6]), int(sys.argv[7])
TLIST = [float(x) for x in sys.argv[8].split(',')] if len(sys.argv) > 8 else None
os.makedirs(od, exist_ok=True)
FPS = 25
NF = int(EV('R_NF', '493'))                      # 48 beats at 146 BPM = 19.73 s -> 493 frames
T = NF / FPS
common.BEAT = T / 48
os.environ.setdefault('LOOPQ', str(T))
TURNS = EV('R_TURNS', '4')                       # one turn per 12 beats (= the previous rose)
# ---------------- mesh ----------------
CACHE = '/tmp/redrose_mesh.npz'
if os.path.exists(CACHE):
    z = np.load(CACHE); V, VT, F, FT = z['V'], z['VT'], z['F'], z['FT']
else:
    V, VT, F, FT = [], [], [], []
    for ln in open(objp):
        if ln.startswith('v '):
            V.append(ln.split()[1:4])
        elif ln.startswith('vt '):
            VT.append(ln.split()[1:3])
        elif ln.startswith('f '):
            p = [q.split('/') for q in ln.split()[1:4]]
            F.append([int(q[0]) - 1 for q in p]); FT.append([int(q[1]) - 1 for q in p])
    V, VT, F, FT = np.array(V, float), np.array(VT, float), np.array(F), np.array(FT)
    np.savez(CACHE, V=V, VT=VT, F=F, FT=FT)
Vc = V - V.mean(0)                                # OBJ is Y-up already
Vc /= np.ptp(Vc[:, 1]); Vc[:, 1] -= (Vc[:, 1].min() + Vc[:, 1].max()) / 2
top_ = Vc[:, 1] > np.percentile(Vc[:, 1], 80)                         # turntable axis through the bloom
Vc[:, 0] -= np.median(Vc[top_, 0]); Vc[:, 2] -= np.median(Vc[top_, 2])
col = cv2.imread(colp).astype(np.float32) / 255
nrm = cv2.imread(nrmp).astype(np.float32) / 255
aom = cv2.imread(aop, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255
N = int(EV('R_N', '1600000'))
A_, B_, C_ = Vc[F[:, 0]], Vc[F[:, 1]], Vc[F[:, 2]]
e1, e2 = B_ - A_, C_ - A_
fn = np.cross(e1, e2); area = np.linalg.norm(fn, axis=1) / 2 + 1e-12; fn = fn / (2 * area[:, None])
# tangent frame per face (for the normal map)
d1 = VT[FT[:, 1]] - VT[FT[:, 0]]; d2 = VT[FT[:, 2]] - VT[FT[:, 0]]
den = d1[:, 0] * d2[:, 1] - d2[:, 0] * d1[:, 1]; den = np.where(np.abs(den) < 1e-12, 1e-12, den)
tg = (e1 * d2[:, 1:2] - e2 * d1[:, 1:2]) / den[:, None]
tg = tg - fn * (tg * fn).sum(1, keepdims=True); tg /= np.linalg.norm(tg, axis=1, keepdims=True) + 1e-12
bt = np.cross(fn, tg)
rng = np.random.default_rng(1)
fi = rng.choice(len(F), N, p=area / area.sum())
r1, r2 = rng.random(N), rng.random(N); sq = np.sqrt(r1)
w0, w1, w2 = 1 - sq, sq * (1 - r2), sq * r2
P = (A_[fi] * w0[:, None] + B_[fi] * w1[:, None] + C_[fi] * w2[:, None]).astype(np.float32)
UV = VT[FT[fi, 0]] * w0[:, None] + VT[FT[fi, 1]] * w1[:, None] + VT[FT[fi, 2]] * w2[:, None]
th_, tw_ = col.shape[:2]
ty_ = np.clip(((1 - UV[:, 1]) * th_).astype(int), 0, th_ - 1); tx_ = np.clip((UV[:, 0] * tw_).astype(int), 0, tw_ - 1)
alb = col[ty_, tx_]; ao = aom[ty_, tx_]
nt = nrm[ty_, tx_][:, ::-1] * 2 - 1                                    # BGR -> RGB, tangent space
nt[:, 1] *= EV('R_NMY', '1')                                           # (flip if the map is DirectX-style)
NS = EV('R_NMS', '1.0')
Nn = fn[fi] * nt[:, 2:3] + NS * (tg[fi] * nt[:, 0:1] + bt[fi] * nt[:, 1:2])
Nn = (Nn / (np.linalg.norm(Nn, axis=1, keepdims=True) + 1e-9)).astype(np.float32)
N0 = fn[fi].astype(np.float32)                                         # geometric normal (ripple direction)
lumA = alb @ np.array([EV('R_WB', '0.08'), EV('R_WG', '0.37'), EV('R_WR', '0.55')], np.float32)
red = np.clip((alb[:, 2] - 0.5 * (alb[:, 0] + alb[:, 1])) / 0.4, 0, 1)    # petals vs leaves (for brightness balance)
al = np.clip((lumA - 0.05) / 0.5, 0, 1) ** 0.75
al = al * (EV('R_LEAF', '0.75') + (1 - EV('R_LEAF', '0.75')) * red) + EV('R_PET', '0.25') * red   # bloom reads first
al = (al * (EV('R_AO0', '0.35') + (1 - EV('R_AO0', '0.35')) * ao)).astype(np.float32)
del A_, B_, C_, UV, alb, nt, e1, e2
q1, q2, q3, q4, q5, q6 = (rng.random(N).astype(np.float32) for _ in range(6))
hgt = ((P[:, 1] - P[:, 1].min()) / np.ptp(P[:, 1])).astype(np.float32)
rad0 = np.hypot(P[:, 0], P[:, 2]).astype(np.float32); ang0 = np.arctan2(P[:, 2], P[:, 0]).astype(np.float32)
# smooth 3D noise: overlapping, wave-like front (layers peel at different moments)
_g = np.random.default_rng(7); nz = np.zeros(N, np.float32)
for _k in range(6):
    d_ = _g.normal(size=3); d_ /= np.linalg.norm(d_)
    nz += np.sin(P @ d_ * EV('R_NFREQ', '9') * (0.7 + 0.6 * _g.random()) + 6.28 * _g.random()).astype(np.float32)
nz = nz / 6 ** 0.5
key = np.clip(hgt + EV('R_NOISE', '0.22') * nz + EV('R_KJIT', '0.16') * (q1 - 0.5), -0.2, 1.2)   # ragged, overlapping front
# ---------------- timeline (fractions of T) ----------------
# out: each particle peels off its home and spirals UP around the rose (widening), glinting, and evaporates near the
# top; in: a twin flight from below its home spirals up and lands on it. Dissolve and rebuild are two waves
# travelling up the rose (stem foot first) -> particles always swirl bottom -> top.
TD0, TDS = EV('R_TD0', '0.24'), EV('R_TDS', '0.22')       # dissolve: start, spread over the height
TR0, TRS = EV('R_TR0', '0.56'), EV('R_TRS', '0.24')       # rebuild (landing times)
DO, DI = EV('R_DOUT', '0.16'), EV('R_DIN', '0.16')        # flight lengths
td = (TD0 + TDS * key) * T; tr = (TR0 + TRS * key) * T
dout = DO * T * (0.8 + 0.4 * q2); din = DI * T * (0.8 + 0.4 * q3)
YT = EV('R_YTOP', '0.56'); YB = -0.56
RF = EV('R_RF', '0.30')                                   # helix radius (kept inside the turn's envelope)
climb_o = ((YT - P[:, 1]) * (EV('R_CLO', '0.30') + 0.45 * q4)).astype(np.float32)      # never past the apex
climb_i = ((P[:, 1] - YB) * (EV('R_CLI', '0.25') + 0.40 * q5)).astype(np.float32)      # never below the foot
SPIN = EV('R_SPIN', '1.1')                                 # turns around the stem per flight
spin_o = (SPIN * (0.6 + 0.8 * q5)).astype(np.float32); spin_i = (SPIN * (0.6 + 0.8 * q4)).astype(np.float32)
rad_f = (RF * np.tanh((rad0 * EV('R_RX', '1.15') + EV('R_RA', '0.05') + EV('R_RB', '0.10') * q3) / RF)).astype(np.float32)   # soft cap: no wall   # the cloud keeps the rose's shape


def flight(tt):
    """returns r, angle, y (model frame), b (0 home .. 1 fully airborne), s (flight progress), vis (spark weight)"""
    so = (tt - td) / dout; si = (tt - (tr - din)) / din
    out_ = (so > 0) & (so < 1); in_ = (si > 0) & (si < 1) & ~out_
    gone = (so >= 1) & (si <= 0)                                     # between the flights: evaporated
    s = np.where(out_, so, np.where(in_, 1 - si, 0)).astype(np.float32)   # 0 at home .. 1 far end (top / bottom)
    s = np.clip(s, 0, 1)
    se = s * s * (3 - 2 * s) * 0.5 + 0.5 * s
    b = np.clip(s / 0.18, 0, 1); b = b * b * (3 - 2 * b)
    spin = np.where(out_, spin_o, spin_i); climb = np.where(out_, climb_o, -climb_i)
    th = ang0 + EV('R_DIR', '-1') * 2 * np.pi * spin * se * np.where(out_, 1, -1)
    rr = rad0 + (rad_f - rad0) * np.sin(np.pi * 0.5 * np.minimum(se * 1.6, 1))
    yy = P[:, 1] + climb * se + EV('R_YSP', '0.07') * (q6 - 0.5) * se          # the cloud loosens vertically too
    wob = 0.010 * np.sin(2 * np.pi * (2.5 * s + q6))
    r = (1 - b) * rad0 + b * (rr + wob)
    a = (1 - b) * ang0 + b * th
    y = (1 - b) * P[:, 1] + b * yy
    air = out_ | in_
    vis = np.where(out_, np.clip((1 - so) / 0.45, 0, 1) ** 1.2, np.clip(si / 0.35, 0, 1) ** 1.2) * air
    b = np.where(gone, 1.0, b * air)
    return r, a, y, b, s, vis


def ripple(tt):
    """surface shiver just ahead of the dissolve front (and a faint one behind the rebuild), plus a slow breath"""
    fr_d = ((tt / T) - TD0) / TDS                                       # front position in key units
    fr_r = ((tt / T) - TR0) / TRS
    g_d = np.exp(-((key - fr_d - 0.07) / 0.07) ** 2)
    g_r = np.exp(-((key - fr_r + 0.05) / 0.06) ** 2) * 0.6
    breath = 0.25 * np.sin(2 * np.pi * (2 * tt / T - 1.5 * hgt))
    amp = EV('R_RIP', '0.010') * ((g_d + g_r) * np.sin(2 * np.pi * (6 * key - 8 * tt / T * 8)) + 0.15 * breath)
    return amp.astype(np.float32)


# ---------------- camera + framing (fit over the whole turn + the fountain) ----------------
tilt = np.radians(EV('R_TILT', '6')); Dcam = EV('R_DIST', '6.0')
Rx = np.array([[1, 0, 0], [0, np.cos(tilt), -np.sin(tilt)], [0, np.sin(tilt), np.cos(tilt)]], np.float32)
Rmax = max(float(np.percentile(rad0, 99.9)), RF + 0.02)
ys = np.linspace(-0.5, 0.5, 9)
cyl = np.array([[Rmax * np.cos(a), y, Rmax * np.sin(a)] for a in np.linspace(0, 2 * np.pi, 72, endpoint=False) for y in np.r_[ys, YT * 1.04 + 0.02, -YT * 1.04 - 0.02]], np.float32)
qc = cyl @ Rx.T
xs_ = qc[:, 0] * Dcam / (Dcam - qc[:, 2]); ys_ = qc[:, 1] * Dcam / (Dcam - qc[:, 2])
cy_ = (ys_.max() + ys_.min()) / 2
MX, MY = EV('R_MX', '0.04'), EV('R_MY', '0.035')                     # safe margins (each side)
fcam = min(W * (1 - 2 * MX) / np.ptp(xs_), H * (1 - 2 * MY) / np.ptp(ys_))
print('fit: Rmax %.3f, rose height %.0f px of %d (%.0f%%), width use %.0f%%' % (Rmax, fcam * 1.0, H, 100 * fcam / H, 100 * fcam * np.ptp(xs_) / W), flush=True)
nv = lambda *a: np.array(a, np.float32) / np.linalg.norm(a)
Lk = nv(-0.55, 0.55, 0.62); Lb = nv(0.4, 0.35, -0.85)


def project(Pw, yaw):
    Ry = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]], np.float32)
    R = Rx @ Ry
    Pc = Pw @ R.T
    Zc = Dcam - Pc[:, 2]
    return W / 2 + fcam * Pc[:, 0] * Dcam / Zc, H / 2 - fcam * (Pc[:, 1] * Dcam / Zc - cy_), Zc, Pc, R


LREF = {}
if os.environ.get('R_LREF'):
    LREF['lref'] = EV('R_LREF', '1')


def frame(t):
    tt = t % T
    yaw = np.radians(EV('R_YAW', '20')) + EV('R_DIR', '-1') * 2 * np.pi * TURNS * tt / T
    r, a, y, b, s, fvis = flight(tt)
    air = fvis > 0
    rp = ripple(tt)
    home = b < 0.02
    Pw = np.stack([r * np.cos(a), y, r * np.sin(a)], -1).astype(np.float32)
    Pw[home] += N0[home] * rp[home, None]
    u, v, Zc, Pc, R = project(Pw, yaw)
    Nc = Nn @ R.T
    view = -np.stack([Pc[:, 0], Pc[:, 1], Pc[:, 2] - Dcam], -1); view /= np.linalg.norm(view, axis=1, keepdims=True)
    flip = (Nc * view).sum(1) < 0; Nc[flip] *= -1
    lam = np.clip(Nc @ Lk, 0, 1); back = np.clip(Nc @ Lb, 0, 1); ndv = np.clip((Nc * view).sum(1), 0, 1)
    hh = Lk + view; hh /= np.linalg.norm(hh, axis=1, keepdims=True)
    spec = np.clip((Nc * hh).sum(1), 0, 1) ** 28
    lum_p = al * (EV('R_FILL', '0.10') + EV('R_KEY', '0.9') * lam ** 1.2) + EV('R_RIM', '0.7') * (1 - ndv) ** 2.5 * (0.4 + 0.6 * back) + 0.45 * spec * al
    # ---- resting surface -> z-buffer -> picture + depth ----
    ui = np.round(u).astype(int); vi = np.round(v).astype(int)
    ok = home & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
    idx = vi[ok] * W + ui[ok]; zz = Zc[ok]
    order = np.lexsort((zz, idx)); idx_s = idx[order]
    first = np.r_[True, idx_s[1:] != idx_s[:-1]] if idx_s.size else np.zeros(0, bool)
    sel = np.nonzero(ok)[0][order[first]]
    zbuf = np.full(H * W, np.inf, np.float32); zbuf[idx_s[first]] = Zc[sel]
    lbuf = np.zeros(H * W, np.float32); lbuf[idx_s[first]] = lum_p[sel]
    zbuf = zbuf.reshape(H, W); lbuf = lbuf.reshape(H, W)
    cov = np.isfinite(zbuf).astype(np.uint8)
    zf = np.where(cov > 0, zbuf, 0).astype(np.float32); lf = np.where(cov > 0, lbuf, 0).astype(np.float32)
    ws = cv2.blur(cov.astype(np.float32), (3, 3)) + 1e-6
    zb = np.where(cov > 0, zf, cv2.blur(zf, (3, 3)) / ws); lb = np.where(cov > 0, lf, cv2.blur(lf, (3, 3)) / ws)
    m = cv2.morphologyEx(cov, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)).astype(np.float32)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    zocc = np.where(m > 0, zb, np.inf).astype(np.float32)
    zlo, zhi = Dcam - 0.4, Dcam + 0.4
    dep = np.where(m > 0, 0.25 + 0.75 * np.clip((zhi - zb) / (zhi - zlo), 0, 1), 0).astype(np.float32)
    if 'lref' not in LREF:
        LREF['lref'] = float(np.percentile(lb[m > 0], 99.3)); print('lref', LREF['lref'], flush=True)
    lum = (lb * m / LREF['lref']).astype(np.float32)
    DS.PRESMAP['a'] = np.clip((cv2.GaussianBlur(m, (0, 0), DS.PITCH_PX * 0.6) - 0.45) / 0.45, 0, 1).astype(np.float32)
    # ---- particles in the air: sparks (motion-blurred over the shutter), hidden behind the resting rose ----
    fl = np.nonzero(air & (b > 0.02) & (q6 < EV('R_SPK', '0.07')))[0]
    U, Vv, Wt, Ci = [], [], [], []
    K = int(EV('R_MBK', '5'))
    lv = np.clip(lum_p[fl] / LREF['lref'], 0, 1)
    for k in range(K):
        tk = tt + (k + 0.5) / K * EV('R_SHUT', '1.4') / FPS
        rk, ak, yk, bk, sk, vk_ = flight(tk)
        Pf = np.stack([rk[fl] * np.cos(ak[fl]), yk[fl], rk[fl] * np.sin(ak[fl])], -1).astype(np.float32)
        yawk = np.radians(EV('R_YAW', '20')) + EV('R_DIR', '-1') * 2 * np.pi * TURNS * tk / T
        uk, vk, zk, _, _ = project(Pf, yawk)
        ins = (uk > 0) & (uk < W - 1) & (vk > 0) & (vk < H - 1)
        hid = np.zeros(fl.size, bool)
        hid[ins] = zk[ins] > zocc[vk[ins].astype(int), uk[ins].astype(int)] + 0.02
        vis = ins & ~hid
        near = np.clip((Dcam + 0.35 - zk) / 0.7, 0.35, 1.25)           # depth: near brighter
        env_ = vk_[fl]
        tw_ = 1 + EV('R_TWK', '2.2') * np.maximum(np.sin(2 * np.pi * (int(EV('R_TWF', '40')) * tk / T + 7 * q4[fl])), 0) ** 10
        wv = (0.3 + 0.7 * lv) * env_ * near * EV('R_SPKW', '0.55') / K * tw_
        U.append(uk[vis]); Vv.append(vk[vis]); Wt.append(wv[vis]); Ci.append((150 + 100 * lv + 30 * (tw_ > 1.5))[vis])
    # ---- swirling dust inside the safe frame: slow helices around the rose, rising, whole cycles per loop ----
    if 'dust' not in LREF:
        g_ = np.random.default_rng(11); nd = int(EV('R_ND', '900'))
        LREF['dust'] = (g_.uniform(0, 2 * np.pi, nd), g_.uniform(0.6, 1.0, nd) * Rmax, g_.random(nd), g_.integers(1, 4, nd),
                        g_.uniform(0.05, 0.5, nd) ** 1.5, g_.uniform(150, 225, nd), g_.integers(2, 7, nd), g_.random(nd))
    a0_, rr_, ph_, nup_, dw_, dc_, nsp_, tq_ = LREF['dust']
    yy_ = ((ph_ + nup_ * tt / T) % 1.0) * 1.1 - 0.55                     # rising, wraps under the frame safe area
    aa_ = a0_ + EV('R_DIR', '-1') * 2 * np.pi * nsp_ * tt / T
    Pd = np.stack([rr_ * np.cos(aa_), yy_, rr_ * np.sin(aa_)], -1).astype(np.float32)
    ud, vd, zd, _, _ = project(Pd, 0.0)
    fade = np.sin(np.pi * np.clip((yy_ + 0.55) / 1.1, 0, 1)) ** 0.8         # fade in at the foot, out at the top
    insd = (ud > 0) & (ud < W - 1) & (vd > 0) & (vd < H - 1)
    hidd = np.zeros(ud.size, bool); hidd[insd] = zd[insd] > zocc[vd[insd].astype(int), ud[insd].astype(int)] + 0.02
    twd = 0.5 + 0.5 * np.sin(2 * np.pi * (3 * tt / T * 4 + tq_)) ** 2
    U.append(ud[~hidd]); Vv.append(vd[~hidd]); Wt.append((dw_ * EV('R_DUST', '0.45') * fade * twd)[~hidd]); Ci.append(dc_[~hidd])
    U, Vv, Wt, Ci = (np.concatenate(a_) for a_ in (U, Vv, Wt, Ci))
    DS.EXTRA.update(u=U, v=Vv, w=Wt, col=L.PAL[np.clip(Ci, 0, 255).astype(np.uint8)])
    return DS.render(dep, 1, tt, None, None, None, None, lum)


frame(0.0)                                         # exposure reference from the intact rose
if TLIST:
    for tq in TLIST:
        img = frame(tq * T)
        cv2.imwrite(f'{od}/rose916_t{int(tq * 1000):04d}.png', (np.clip(img, 0, 1) * 65535 + 0.5).astype(np.uint16)); print('t', tq, flush=True)
else:
    for fr in range(f0, f1):
        img = frame(fr / FPS)
        cv2.imwrite(f'{od}/rose916_{fr:04d}.png', (np.clip(img, 0, 1) * 65535 + 0.5).astype(np.uint16)); print(fr, flush=True)
