"""3D rose loop: turntable rotation + a wave of particles travelling up the rose. Ahead of the wave the surface peels
off and spirals up and around the stem (with the rotation), behind it the particles land back (staggered).
Resting surface -> point-splat z-buffer -> picture + depth -> depth_sheet.render (the bouquet look);
flying particles -> glittering sparks (DS.EXTRA), hidden behind the resting surface. Every motion repeats at T.
usage: python3 rose3d_loop.py rosa.obj albedo.jpg outdir f0 f1          env: preset rose3d_sf.env + R_T, R_NW ..."""
import sys, os
import numpy as np, cv2
import depth_sheet as DS
import sheet_loop2 as L
from common import W, H

EV = lambda k, d: float(os.environ.get(k, d))
objp, texp, od, f0, f1 = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
os.makedirs(od, exist_ok=True)
T = EV('R_T', '9.84')                                   # loop length: 246 frames = 24 beats (seamless at frame level)
import common
common.BEAT = T / 24                                     # beat pulse repeats exactly too
os.environ.setdefault('LOOPQ', str(T))                   # glint rhythms: whole cycles per loop
NW = EV('R_NW', '2')                                    # waves per loop
FPS = 25
z = np.load('/tmp/rosa_mesh.npz') if os.path.exists('/tmp/rosa_mesh.npz') else None
if z is None:
    V, VT, F, FT = [], [], [], []
    for ln in open(objp):
        if ln.startswith('v '):
            V.append([float(x) for x in ln.split()[1:4]])
        elif ln.startswith('vt '):
            VT.append([float(x) for x in ln.split()[1:3]])
        elif ln.startswith('f '):
            p = [q.split('/') for q in ln.split()[1:4]]
            F.append([int(q[0]) - 1 for q in p]); FT.append([int(q[1]) - 1 for q in p])
    V, VT, F, FT = (np.array(a) for a in (V, VT, F, FT))
    np.savez('/tmp/rosa_mesh.npz', V=V, VT=VT, F=F, FT=FT)
else:
    V, VT, F, FT = z['V'], z['VT'], z['F'], z['FT']
tex = cv2.imread(texp).astype(np.float32) / 255
# model: Z up -> y up, centred, height 1
Vc = V - V.mean(0)
Vc = np.stack([Vc[:, 0], Vc[:, 2], -Vc[:, 1]], -1)
Vc /= np.ptp(Vc[:, 1]); Vc[:, 1] -= (Vc[:, 1].min() + Vc[:, 1].max()) / 2
top_ = Vc[:, 1] > np.percentile(Vc[:, 1], 60)
axis_xz = np.array([np.median(Vc[top_, 0]), np.median(Vc[top_, 2])])   # turntable axis: through the flower head
Vc[:, 0] -= axis_xz[0]; Vc[:, 2] -= axis_xz[1]
N = int(EV('R_N', '3000000'))
A_, B_, C_ = Vc[F[:, 0]], Vc[F[:, 1]], Vc[F[:, 2]]
fn = np.cross(B_ - A_, C_ - A_); area = np.linalg.norm(fn, axis=1) / 2 + 1e-12; fn = fn / (2 * area[:, None])
rng = np.random.default_rng(1)
fi = rng.choice(len(F), N, p=area / area.sum())
r1, r2 = rng.random(N), rng.random(N); sq = np.sqrt(r1)
w0, w1, w2 = 1 - sq, sq * (1 - r2), sq * r2
P = A_[fi] * w0[:, None] + B_[fi] * w1[:, None] + C_[fi] * w2[:, None]
UV = VT[FT[fi, 0]] * w0[:, None] + VT[FT[fi, 1]] * w1[:, None] + VT[FT[fi, 2]] * w2[:, None]
th_, tw_ = tex.shape[:2]
alb = tex[np.clip(((1 - UV[:, 1]) * th_).astype(int), 0, th_ - 1), np.clip((UV[:, 0] * tw_).astype(int), 0, tw_ - 1)]
al = np.clip((alb @ np.array([0.114, 0.587, 0.299], np.float32) - 0.10) / 0.55, 0, 1) ** 0.8
Nn = fn[fi]
del A_, B_, C_, UV, alb
# per-particle randomness + wave key (height along the rose, 0 = stem foot .. 1 = top of the head)
q1, q2, q3, q4, q5 = (rng.random(N).astype(np.float32) for _ in range(5))
hgt = (P[:, 1] - P[:, 1].min()) / np.ptp(P[:, 1])
rad0 = np.hypot(P[:, 0], P[:, 2]); ang0 = np.arctan2(P[:, 2], P[:, 0])
# whole patches peel off (so the holes read), a share stays home: the rose never vanishes
_g = np.random.default_rng(7); _nz = np.zeros(N, np.float32)
for _k in range(6):
    d_ = _g.normal(size=3); d_ /= np.linalg.norm(d_)
    _nz += np.sin(P @ d_ * EV('R_PFREQ', '18') * (0.7 + 0.6 * _g.random()) + 6.28 * _g.random()).astype(np.float32)
_nz /= 6 ** 0.5
FLY = (_nz > np.quantile(_nz, 1 - EV('R_FLYFR', '0.6'))) & (q1 < 0.95)
tilt = np.radians(EV('R_TILT', '10')); yaw0 = np.radians(EV('R_YAW', '190'))
Rx = np.array([[1, 0, 0], [0, np.cos(tilt), -np.sin(tilt)], [0, np.sin(tilt), np.cos(tilt)]])
Dcam = EV('R_DIST', '2.2')
qv = Vc @ Rx.T
cy_ = (qv[:, 1].min() + qv[:, 1].max()) / 2
fcam = H * EV('R_FILLH', '0.6') * Dcam / np.ptp(qv[:, 1])
nv = lambda *a: np.array(a) / np.linalg.norm(a)
Lk = nv(-0.5, 0.6, 0.65); Lb = nv(0.35, 0.45, -0.85)


def offsets(t):
    """flight state: local wave phase -> displacement e (0 home .. 1 furthest), spiral around the stem"""
    ph = (t / T * NW - EV('R_HK', '1.0') * hgt - 0.22 * q2) % 1.0          # wave travels up the rose
    FL = EV('R_FL', '0.5')                                                 # share of a wave cycle in the air
    s = np.clip(ph / (FL * (0.8 + 0.4 * q3)), 0, 1)
    e = np.sin(np.pi * s) ** 1.3 * FLY                                      # out and back (staggered landing)
    e = e * (s < 1)
    dth = EV('R_SPIN', '1.5') * e * (0.6 + 0.8 * q4) + 0.6 * e * s           # around the stem, with the rotation
    dr = (EV('R_OUT', '0.10') + 0.25 * rad0) * e * (0.5 + q5)
    dy = EV('R_UPL', '0.14') * e * (0.4 + q3) + 0.05 * np.sin(2 * np.pi * (2 * s + q2)) * e
    return e, dth, dr, dy


def project(Pw, yaw):
    Ry = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])
    Pc = Pw @ (Rx @ Ry).T
    Pc[:, 1] -= cy_
    Zc = Dcam - Pc[:, 2]
    return W / 2 + fcam * Pc[:, 0] / Zc, H / 2 - fcam * Pc[:, 1] / Zc, Zc, Pc, Rx @ Ry


LREF = {}
if os.environ.get('R_LREF'):
    LREF['lref'] = EV('R_LREF', '1')


def frame(t):
    yaw = yaw0 + EV('R_DIR', '-1') * 2 * np.pi * t / T                   # front turns left -> right
    e, dth, dr, dy = offsets(t)
    home = e < 0.02
    u, v, Zc, Pc, R = project(P, yaw)
    Nc = Nn @ R.T
    view = -np.stack([Pc[:, 0], Pc[:, 1], Pc[:, 2] - Dcam], -1); view /= np.linalg.norm(view, axis=1, keepdims=True)
    flip = (Nc * view).sum(1) < 0; Nc[flip] *= -1
    lam = np.clip(Nc @ Lk, 0, 1); back = np.clip(Nc @ Lb, 0, 1); ndv = np.clip((Nc * view).sum(1), 0, 1)
    hh = Lk + view; hh /= np.linalg.norm(hh, axis=1, keepdims=True)
    spec = np.clip((Nc * hh).sum(1), 0, 1) ** 24
    lum_p = al * (EV('R_FILL', '0.10') + EV('R_KEY', '0.9') * lam ** 1.2) + EV('R_RIM', '0.9') * (1 - ndv) ** 2.5 * (0.4 + 0.6 * back) + 0.5 * spec * al
    # ---- resting surface -> z-buffer image ----
    ui = np.round(u).astype(int); vi = np.round(v).astype(int)
    ok = home & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H) & (Zc > 0.1)
    idx = vi[ok] * W + ui[ok]; zz = Zc[ok]
    order = np.lexsort((zz, idx)); idx_s = idx[order]; first = np.r_[True, idx_s[1:] != idx_s[:-1]]
    sel = np.nonzero(ok)[0][order[first]]
    zbuf = np.full(H * W, np.inf, np.float32); zbuf[idx_s[first]] = Zc[sel]
    lbuf = np.zeros(H * W, np.float32); lbuf[idx_s[first]] = lum_p[sel]
    zbuf = zbuf.reshape(H, W); lbuf = lbuf.reshape(H, W)
    cov = np.isfinite(zbuf).astype(np.uint8)
    zf = np.where(cov > 0, zbuf, 0).astype(np.float32); lf = np.where(cov > 0, lbuf, 0).astype(np.float32)
    ws = cv2.blur(cov.astype(np.float32), (3, 3)) + 1e-6
    zb = np.where(cov > 0, zf, cv2.blur(zf, (3, 3)) / ws); lb = np.where(cov > 0, lf, cv2.blur(lf, (3, 3)) / ws)
    m = cv2.morphologyEx(cov, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)).astype(np.float32)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))       # lone leftover specks: no sheet dot
    zlo, zhi = 2.2 - 0.6, 2.2 + 0.6                                          # fixed depth range (no pumping)
    dep = np.where(m > 0, 0.25 + 0.75 * np.clip((zhi - zb) / (zhi - zlo), 0, 1), 0).astype(np.float32)
    if 'lref' not in LREF:                                                    # fixed exposure from the rest pose
        LREF['lref'] = float(np.percentile(lb[m > 0], 99.3))
    lum = (lb * m / LREF['lref']).astype(np.float32)
    DS.PRESMAP['a'] = np.clip((cv2.GaussianBlur(m, (0, 0), DS.PITCH_PX * 0.6) - 0.45) / 0.45, 0, 1).astype(np.float32)
    # ---- particles in flight: a share of them as sparks, motion-blurred over the shutter ----
    fl = np.nonzero(~home & (q5 < EV('R_SPK', '0.06')))[0]
    U, Vv, Wt, Ci = [], [], [], []
    K = int(EV('R_MBK', '6'))
    for k in range(K):
        tk = t + (k + 0.5) / K * EV('R_SHUT', '1.5') / FPS          # long shutter: the spiral reads as trails
        ek, dthk, drk, dyk = offsets(tk)
        a_ = ang0[fl] + dthk[fl]; r_ = rad0[fl] + drk[fl]
        Pf = np.stack([r_ * np.cos(a_), P[fl, 1] + dyk[fl], r_ * np.sin(a_)], -1)
        uk, vk, zk, _, _ = project(Pf, yaw0 + EV('R_DIR', '-1') * 2 * np.pi * tk / T)
        ins = (uk > 0) & (uk < W - 1) & (vk > 0) & (vk < H - 1)
        hid = np.zeros(fl.size, bool)
        hid[ins] = zk[ins] > zb[vk[ins].astype(int), uk[ins].astype(int)] + 0.02     # behind the resting rose
        vis = ins & ~hid
        lv = np.clip(lum_p[fl] / LREF['lref'], 0, 1)
        wv = (0.35 + 0.65 * lv) * np.sin(np.pi * np.clip(ek[fl], 0, 1)) ** 0.5 * EV('R_SPKW', '0.55') / K
        tw_ = 1 + 1.5 * np.maximum(np.sin(2 * np.pi * (3.0 * tk / T * 8 + q4[fl])), 0) ** 8
        U.append(uk[vis]); Vv.append(vk[vis]); Wt.append((wv * tw_)[vis]); Ci.append((150 + 105 * lv)[vis])
    # seamless dust (replaces the renderer's free-drifting specks): slow closed orbits, whole-cycle twinkles
    if 'dust' not in LREF:
        g_ = np.random.default_rng(11); nd = 520
        LREF['dust'] = (g_.uniform(0, W, nd), g_.uniform(0, H, nd), g_.uniform(4, 22, nd), g_.random(nd), g_.integers(1, 6, nd),
                        g_.uniform(0.05, 0.5, nd) ** 1.5, g_.uniform(150, 215, nd))
    du0, dv0, da, dph, dk, dw, dc = LREF['dust']
    ph_ = 2 * np.pi * (t / T + dph)
    U.append(du0 + da * np.cos(ph_)); Vv.append(dv0 + 0.6 * da * np.sin(ph_))
    Wt.append(dw * EV('R_DUST', '0.35') * (0.5 + 0.5 * np.sin(2 * np.pi * dk * t / T + 6.28 * dph) ** 2)); Ci.append(dc)
    U, Vv, Wt, Ci = (np.concatenate(a_) for a_ in (U, Vv, Wt, Ci))
    DS.EXTRA.update(u=U, v=Vv, w=Wt, col=L.PAL[np.clip(Ci, 0, 255).astype(np.uint8)])
    img = DS.render(dep, 1, t, None, None, None, None, lum)
    return img


for fr in range(f0, f1):
    img = frame(fr / FPS)
    cv2.imwrite(f'{od}/rose3d_{fr:04d}.png', (np.clip(img, 0, 1) * 65535 + 0.5).astype(np.uint16))
    print(fr, flush=True)
