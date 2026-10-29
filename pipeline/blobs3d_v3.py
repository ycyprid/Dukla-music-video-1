"""3D halftone sheet v3: horizon in frame (sheet extends to a warm haze), less blur, world-anchored grid
(ready for endless forward travel). Same field, palette and tempo surges as v2.
usage: python3 blobs3d_v3.py still <horizon_fraction> <out.png> [t]   e.g. 0.333 or 0.25"""
import sys
import opensimplex
from common import *

opensimplex.seed(70)


def hexrgb(h):
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


PAL = gradient_lut([(0.00, hexrgb('5A0620')), (0.14, hexrgb('8A0A34')), (0.30, hexrgb('C2183F')),
                    (0.48, hexrgb('D8542C')), (0.64, hexrgb('E8A322')), (0.80, hexrgb('F6B93A')),
                    (0.92, hexrgb('FFDC96')), (1.00, hexrgb('FFF1D8'))])
HAZE = np.array(hexrgb('E39A3C')[::-1], np.float32) / 255          # warm gold haze (BGR)

FPX = H * 1.05
CAMY = 1.05
ZNEAR, ZFAR = 0.9, 20.0
DXW, DZW = 0.032, 0.040                     # world-uniform dot spacing (anchored to the ground)
S = 0.8
RES = 0.035

# world-anchored grid: rows at fixed world z, columns at fixed world x, wide enough for the view frustum
rows = np.arange(ZNEAR, ZFAR, DZW, dtype=np.float32)
pts_x, pts_z = [], []
for z in rows:
    half = z * (W / 2) / FPX * 1.15 + 0.6
    xs = np.arange(-half, half, DXW, dtype=np.float32)
    pts_x.append(xs); pts_z.append(np.full_like(xs, z))
PX = np.concatenate(pts_x); PZ = np.concatenate(pts_z)
UA, UB = PX.min() * S - 3, PX.max() * S + 3
VA, VB = PZ.min() * S - 3, PZ.max() * S + 8
uu = np.arange(UA, UB, RES); vv = np.arange(VA, VB, RES)


def lattice(z, sc=1.0):
    return opensimplex.noise3array(uu * sc, vv * sc, np.array([z]))[0].astype(np.float32)


def sample(lat, u, v):
    n = u.size; pad = (-n) % 1024                                  # remap maps must stay < 32767 wide
    mu = np.pad(((u - UA) / RES).astype(np.float32), (0, pad)).reshape(-1, 1024)
    mv = np.pad(((v - VA) / RES).astype(np.float32), (0, pad)).reshape(-1, 1024)
    return cv2.remap(lat, mu, mv, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT).ravel()[:n]


def field(tau, pulse, travel):
    z = tau * 0.42
    u = PX * S; v = (PZ + travel) * S
    wx = sample(lattice(z * 0.6 + 10, 0.6), u, v)
    wy = sample(lattice(z * 0.6 + 40, 0.6), u, v)
    n = sample(lattice(z), u + 0.9 * wx + tau * 0.06, v + 0.9 * wy)
    lo = 0.02 - 0.06 * pulse
    val = np.clip((n - lo) / (0.62 - lo), 0, 1)
    val = val * val * (3 - 2 * val)
    fold = sample(lattice(z * 0.30 + 70, 0.35), u, v)
    return val, fold


rng = np.random.default_rng(11)
ND = 340
DX = rng.uniform(-3.5, 3.5, ND).astype(np.float32)
DY = rng.uniform(-0.3, 1.2, ND).astype(np.float32)
DZd = rng.uniform(0.3, 7.0, ND).astype(np.float32)
DPH = rng.uniform(0, 6.28, ND).astype(np.float32)


def splat(u, v, w, h, wd):
    img = np.zeros(h * wd, np.float32)
    x0 = np.floor(u).astype(np.int64); y0 = np.floor(v).astype(np.int64)
    fx = (u - x0).astype(np.float32); fy = (v - y0).astype(np.float32)
    for dx, dy, ww in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)), (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
        xi, yi = x0 + dx, y0 + dy
        ok = (xi >= 0) & (xi < wd) & (yi >= 0) & (yi < h)
        img += np.bincount(yi[ok] * wd + xi[ok], weights=(ww * w)[ok], minlength=h * wd)
    return img.reshape(h, wd)


def disc(r):
    k = int(np.ceil(r)) * 2 + 1
    yy, xx = np.mgrid[:k, :k] - k // 2
    rr = np.sqrt(xx * xx + yy * yy)
    d = np.clip(r + 0.5 - rr, 0, 1).astype(np.float32)
    d[rr > r - 0.8] *= 1.25
    return d / d.sum()


LEVELS = [('g', 0.7), ('g', 1.3), ('g', 2.2), ('d', 2.2), ('d', 3.6), ('d', 5.5), ('d', 8.0), ('d', 11.5)]
KER = {L: disc(r) for L, (k, r) in enumerate(LEVELS) if k == 'd'}
RAD = np.array([r if k == 'g' else r * 2 for k, r in LEVELS])
vig = vignette(k=0.45)
APER = 10.0                                  # ~1/3 of v2: most of the sheet sharp
EXP = 1.7


def render(tau, t, pulse, horizon=1 / 3, travel=0.0):
    # aim so the visible far edge of the sheet (haze line) sits at `horizon` of frame height
    pitch = float(np.arctan((0.5 - horizon) * H / FPX) + np.arctan(CAMY / 15.5))
    cp, sp = np.cos(pitch), np.sin(pitch)
    zf = CAMY / np.tan(pitch + 0.12) * 1.6                         # focus a little beyond frame centre
    val, fold = field(tau, pulse, travel)
    Y = fold * 0.55 + val * 0.30
    camx = 0.30 * np.sin(t * 0.13)
    yc = Y - CAMY; xc = PX - camx; zc = PZ
    y2 = yc * cp + zc * sp; z2 = -yc * sp + zc * cp
    ok = z2 > 0.15
    u = W / 2 + FPX * xc / np.maximum(z2, 0.15); v = H / 2 - FPX * y2 / np.maximum(z2, 0.15)
    col = PAL[((0.10 + 0.90 * val) * 255).astype(np.uint8)]
    fog = np.clip((z2 - 4.0) / 13.0, 0, 1)[:, None] ** 0.8           # distance -> warm haze
    col = col * (1 - fog * 0.75) + HAZE * fog * 0.75
    w = (0.30 + 0.70 * val) * ok * 2.0 / (1 + (z2 / 2.6) ** 2.0)
    w = w * (1 - np.clip((z2 - 15.0) / 5.0, 0, 1))                  # fade out before the grid ends
    coc = APER * np.abs(1 / np.maximum(z2, 0.15) - 1 / zf)
    size = FPX * 0.004 / np.maximum(z2, 0.15)
    r = np.maximum(size, coc)
    dx = DX + 0.05 * np.sin(t * 0.4 + DPH) - camx
    dy = DY + 0.06 * np.sin(t * 0.3 + DPH * 1.3) + 0.02 * t - CAMY
    dz2 = -dy * sp + DZd * cp; dy2 = dy * cp + DZd * sp
    du = W / 2 + FPX * dx / np.maximum(dz2, 0.15); dv = H / 2 - FPX * dy2 / np.maximum(dz2, 0.15)
    dcoc = 30.0 * np.abs(1 / np.maximum(dz2, 0.15) - 1 / zf) * 1.6     # dust keeps its lens bokeh
    dcol = PAL[np.full(ND, int(0.78 * 255))] * 0.9
    dw = 3.0 * (dz2 > 0.15) * (0.6 + 0.4 * np.sin(DPH + t * 0.5) ** 2)
    U = np.concatenate([u, du]); V = np.concatenate([v, dv])
    C = np.concatenate([col, dcol]); Wt = np.concatenate([w, dw]); Rr = np.concatenate([r, np.maximum(dcoc, 4.0)])
    lvl = np.argmin(np.abs(RAD[:, None] - Rr[None]), 0)
    img = np.zeros((H, W, 3), np.float32)
    for L, (kind, rad) in enumerate(LEVELS):
        m = lvl == L
        if not m.any():
            continue
        if kind == 'g':
            acc = np.stack([splat(U[m], V[m], Wt[m] * C[m, c], H, W) for c in range(3)], -1)
            img += cv2.GaussianBlur(acc, (0, 0), rad) * (rad / 0.7) ** 1.2
        else:
            acc = np.stack([splat(U[m] / 2, V[m] / 2, Wt[m] * C[m, c], H // 2, W // 2) for c in range(3)], -1)
            img += cv2.resize(cv2.filter2D(acc, -1, KER[L]) * (rad / 2.2) ** 1.2 * 1.6, (W, H))
    img *= EXP
    # faint warm glow along the horizon line
    hy = H / 2 - FPX * np.tan(pitch - np.arctan(CAMY / 15.5))
    yy = np.arange(H, dtype=np.float32)[:, None, None]
    img += HAZE * 0.10 * np.exp(-((yy - hy) / (H * 0.05)) ** 2)
    gsm = cv2.GaussianBlur(cv2.resize(img, (W // 4, H // 4), interpolation=cv2.INTER_AREA), (0, 0), 3)
    img = img + cv2.resize(np.clip(gsm - 0.25, 0, None), (W, H)) * (0.6 + 0.2 * pulse)
    img = img * vig
    L = luma(img)[..., None]
    Lt = L / (1 + L * 0.85)
    return np.clip(img * (Lt / np.maximum(L, 1e-5)) * 1.25, 0, 1)


if __name__ == '__main__':
    if sys.argv[1] == 'still':
        hz = float(sys.argv[2]); out = sys.argv[3]
        t = float(sys.argv[4]) if len(sys.argv) > 4 else 5.0
        # rebuild tau exactly as the animation would at time t
        tau = 0.0
        for i in range(int(t * FPS)):
            tau += (1.0 + 1.6 * soft_pulse(i / FPS)) / FPS
        img = render(tau, t, soft_pulse(t), hz)
        cv2.imwrite(out, (img * 255 + 0.5).astype(np.uint8))
        print('points', PX.size)
