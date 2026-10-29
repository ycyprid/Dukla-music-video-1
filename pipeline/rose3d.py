"""3D rose model (rosa.obj + texture) in the dot-sheet look.
The mesh is rendered with a small point-splat z-buffer (dense surface samples, area-weighted) into the same inputs
the footage shots use -- a picture (lum) and a depth map -- then drawn by depth_sheet.render exactly like the rose
bouquet (picture mode, halftone discs, glow, glints).
usage: python3 rose3d.py rosa.obj albedo.jpg out.png [t]      env: R_YAW (deg), R_TILT (deg), R_ZOOM, R_N (samples)"""
import sys, os
import numpy as np, cv2
import depth_sheet as DS
from common import W, H

EV = lambda k, d: float(os.environ.get(k, d))
objp, texp, out = sys.argv[1], sys.argv[2], sys.argv[3]
t = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
CACHE = '/tmp/rosa_mesh.npz'
if os.path.exists(CACHE):
    z = np.load(CACHE); V, VT, F, FT = z['V'], z['VT'], z['F'], z['FT']
else:
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
    np.savez(CACHE, V=V, VT=VT, F=F, FT=FT)
tex = cv2.imread(texp).astype(np.float32) / 255
# ---- normalise: centre, stem vertical (principal axis -> y), height 1 ----
c = V.mean(0); Vc = V - c
ev, evec = np.linalg.eigh(np.cov(Vc.T))
ax = evec[:, np.argmax(ev)]
if (Vc @ ax)[np.argmax(np.abs(Vc @ ax))] < 0:                   # flower head up: decided below by width
    pass
up = np.array([float(x_) for x_ in os.environ['R_UP'].split(',')]) if os.environ.get('R_UP') else (np.array([0, 1.0, 0]) if os.environ.get('R_NOAXIS') else ax)
# rotate so `up` -> +y
v_ = np.cross(up, [0, 1, 0]); s_ = np.linalg.norm(v_); c_ = up @ [0, 1, 0]
if s_ > 1e-6:
    K = np.array([[0, -v_[2], v_[1]], [v_[2], 0, -v_[0]], [-v_[1], v_[0], 0]])
    Rm = np.eye(3) + K + K @ K * ((1 - c_) / s_ ** 2)
    Vc = Vc @ Rm.T
# flower head (the wide end) at the top
y = Vc[:, 1]; top = y > np.percentile(y, 70); bot = y < np.percentile(y, 30)
if not os.environ.get('R_UP') and np.ptp(Vc[top][:, [0, 2]], 0).sum() < np.ptp(Vc[bot][:, [0, 2]], 0).sum():
    Vc = Vc * [1, -1, -1]
Vc = Vc / np.ptp(Vc[:, 1])
Vc[:, 1] -= Vc[:, 1].mean()
# ---- area-weighted surface samples (cached per N) ----
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
Nn = fn[fi]
# ---- camera: turntable yaw about the stem, slight tilt down ----
yaw = np.radians(EV('R_YAW', '35')); tilt = np.radians(EV('R_TILT', '12'))
Ry = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])
Rx = np.array([[1, 0, 0], [0, np.cos(tilt), -np.sin(tilt)], [0, np.sin(tilt), np.cos(tilt)]])
R = Rx @ Ry
Pc = P @ R.T; Nc = Nn @ R.T
Dcam = EV('R_DIST', '2.2')
# auto-frame: centre the model's rest pose and fill R_FILLH of the frame height (stable across a turntable: use yaw 0 bbox)
q = (Vc @ Rx.T) if os.environ.get('R_FRAME_REST', '1') == '1' else (Vc @ R.T)
cx_, cy_ = (q[:, 0].min() + q[:, 0].max()) / 2, (q[:, 1].min() + q[:, 1].max()) / 2
Pc[:, 0] += -cx_ + EV('R_DX', '0'); Pc[:, 1] += -cy_ + EV('R_DY', '0')
Zc = Dcam - Pc[:, 2]
f = H * EV('R_FILLH', '0.86') * Dcam / max(np.ptp(q[:, 1]), np.ptp(q[:, 0]) * H / W * 1.0)
u = W / 2 + f * Pc[:, 0] / Zc; v = H / 2 - f * Pc[:, 1] / Zc
# two-sided: normals face the camera
view = -np.stack([Pc[:, 0], Pc[:, 1], Pc[:, 2] - Dcam], -1); view /= np.linalg.norm(view, axis=1, keepdims=True)
flip = (Nc * view).sum(1) < 0; Nc[flip] *= -1
# ---- lighting: soft key upper-left-front, strong backlight rim, gentle fill ----
nv = lambda *a: np.array(a) / np.linalg.norm(a)
Lk = nv(-0.5, 0.6, 0.65); Lb = nv(0.35, 0.45, -0.85)
lam = np.clip(Nc @ Lk, 0, 1); back = np.clip(Nc @ Lb, 0, 1)
ndv = np.clip((Nc * view).sum(1), 0, 1)
rim = (1 - ndv) ** 2.5
hh = Lk + view; hh /= np.linalg.norm(hh, axis=1, keepdims=True)
spec = np.clip((Nc * hh).sum(1), 0, 1) ** 24
al = alb @ np.array([0.114, 0.587, 0.299], np.float32)                 # albedo luminance
al = np.clip((al - 0.10) / 0.55, 0, 1) ** 0.8
lum_p = al * (EV('R_FILL', '0.18') + EV('R_KEY', '0.75') * lam ** 1.2) + EV('R_RIM', '0.9') * rim * (0.4 + 0.6 * back) + 0.5 * spec * al
# ---- point-splat z-buffer at design res ----
ui = np.round(u).astype(int); vi = np.round(v).astype(int)
ok = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H) & (Zc > 0.1)
idx = vi[ok] * W + ui[ok]; zz = Zc[ok]
order = np.lexsort((zz, idx))                                          # nearest first per pixel
idx_s = idx[order]; first = np.r_[True, idx_s[1:] != idx_s[:-1]]
sel = np.nonzero(ok)[0][order[first]]
zbuf = np.full(H * W, np.inf, np.float32); zbuf[idx_s[first]] = Zc[sel]
lbuf = np.zeros(H * W, np.float32); lbuf[idx_s[first]] = lum_p[sel]
zbuf = zbuf.reshape(H, W); lbuf = lbuf.reshape(H, W)
cov = np.isfinite(zbuf).astype(np.uint8)
# fill splat holes (small gaps between samples)
k3 = np.ones((3, 3), np.uint8)
covc = cv2.morphologyEx(cov, cv2.MORPH_CLOSE, k3)
zf = np.where(cov > 0, zbuf, 0).astype(np.float32); lf = np.where(cov > 0, lbuf, 0).astype(np.float32)
wsum = cv2.blur(cov.astype(np.float32), (3, 3)) + 1e-6
zb = np.where(cov > 0, zf, cv2.blur(zf, (3, 3)) / wsum); lb = np.where(cov > 0, lf, cv2.blur(lf, (3, 3)) / wsum)
m = covc.astype(np.float32)
zmin, zmax = zb[m > 0].min(), zb[m > 0].max()
dep = np.where(m > 0, 0.25 + 0.75 * (zmax - zb) / (zmax - zmin + 1e-6), 0).astype(np.float32)
lum = (lb * m).astype(np.float32)
lum = lum / (np.percentile(lum[m > 0], 99.3) + 1e-6)
DS.PRESMAP['a'] = np.clip((cv2.GaussianBlur(m, (0, 0), DS.PITCH_PX * 0.6) - 0.45) / 0.45, 0, 1).astype(np.float32)
img = DS.render(dep, 1, t, None, None, None, None, lum)
cv2.imwrite(out, (np.clip(img, 0, 1) * 65535 + 0.5).astype(np.uint16))
cv2.imwrite(out.replace('.png', '_lum.jpg'), (np.clip(lum, 0, 1) * 255).astype(np.uint8))
print('done', out, 'coverage px', int(m.sum()))
