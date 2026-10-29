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
DEEP = np.array(hexrgb('4A0418')[::-1], np.float32) / 255          # distance recedes into deep burgundy

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


def field_raw(tau, wz):
    """raw noise at world rows PZ + wz (wz snapped to the row grid, so dots stay ground-anchored)"""
    z = tau * 0.42
    u = PX * S; v = (PZ + wz) * S
    wx = sample(lattice(z * 0.6 + 10, 0.6), u, v)
    wy = sample(lattice(z * 0.6 + 40, 0.6), u, v)
    n = sample(lattice(z), u + 0.9 * wx + tau * 0.06, v + 0.9 * wy)
    fold = sample(lattice(z * 0.30 + 70, 0.35), u, v)
    return n, fold


def shape(n, pulse):
    lo = 0.02 - 0.06 * pulse
    val = np.clip((n - lo) / (0.62 - lo), 0, 1)
    return val * val * (3 - 2 * val)


LOOP = 15.0                                   # 15 s = exactly 25 beats at 100 BPM
rng = np.random.default_rng(11)
ND = 340
DX = rng.uniform(-3.5, 3.5, ND).astype(np.float32)
DY = rng.uniform(-0.3, 1.2, ND).astype(np.float32)
DZMIN, DZRANGE = 0.3, 7.5
DZ0 = rng.uniform(0, DZRANGE, ND).astype(np.float32)
DPH = rng.uniform(0, 6.28, ND).astype(np.float32)
DK = rng.integers(1, 3, ND).astype(np.float32)          # sway cycles per loop (1 or 2) -> periodic


def dust_z(t):
    """each particle travels toward camera and wraps exactly once per loop (seamless)"""
    zz = np.mod(DZ0 - DZRANGE * t / LOOP, DZRANGE)
    fade = np.clip(zz / 0.6, 0, 1) * np.clip((DZRANGE - zz) / 1.2, 0, 1)   # invisible at the wrap
    return DZMIN + zz, fade


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
NEAR_GAIN = 9.0
GLOW = 1.0
EXP = 1.7


def render(val, fold, frac, t, pulse, horizon=0.25):
    # aim so the visible far edge of the sheet (haze line) sits at `horizon` of frame height
    pitch = float(np.arctan((0.5 - horizon) * H / FPX) + np.arctan(CAMY / 15.5))
    cp, sp = np.cos(pitch), np.sin(pitch)
    zf = CAMY / np.tan(pitch + 0.12) * 1.6                         # focus a little beyond frame centre
    Y = fold * 0.55 + val * 0.30
    camx = 0.30 * np.sin(2 * np.pi * t / LOOP)                    # one sway cycle per loop
    yc = Y - CAMY; xc = PX - camx; zc = PZ - frac                 # ground glides toward camera
    y2 = yc * cp + zc * sp; z2 = -yc * sp + zc * cp
    ok = z2 > 0.15
    u = W / 2 + FPX * xc / np.maximum(z2, 0.15); v = H / 2 - FPX * y2 / np.maximum(z2, 0.15)
    col = PAL[((0.10 + 0.90 * val) * 255).astype(np.uint8)]
    fog = np.clip((z2 - 2.5) / 12.0, 0, 1)[:, None] ** 0.8           # distance -> recedes into deep burgundy
    col = col * (1 - fog * 0.7) + DEEP * fog * 0.7
    # front brightest, back dimmer + more transparent: undo the on-screen pile-up of far dots (~1/z^2),
    # then fade with distance
    w = (0.30 + 0.70 * val) * ok * 2.0 * np.minimum(1.0, (1.3 / np.maximum(z2, 0.15)) ** 2) \
        * np.exp(-np.maximum(z2 - 1.3, 0) / 6.5) * NEAR_GAIN
    w = w * (1 - np.clip((z2 - 15.0) / 5.0, 0, 1))                  # fade out before the grid ends
    coc = APER * np.abs(1 / np.maximum(z2, 0.15) - 1 / zf)
    size = FPX * 0.004 / np.maximum(z2, 0.15)
    r = np.maximum(size, coc)
    DZd, dfade = dust_z(t)
    ph = 2 * np.pi * DK * t / LOOP
    dx = DX + 0.05 * np.sin(ph + DPH) - camx
    dy = DY + 0.06 * np.sin(ph + DPH * 1.3) - CAMY
    dz2 = -dy * sp + DZd * cp; dy2 = dy * cp + DZd * sp
    du = W / 2 + FPX * dx / np.maximum(dz2, 0.15); dv = H / 2 - FPX * dy2 / np.maximum(dz2, 0.15)
    dcoc = 30.0 * np.abs(1 / np.maximum(dz2, 0.15) - 1 / zf) * 1.6     # dust keeps its lens bokeh
    dcol = PAL[np.full(ND, int(0.78 * 255))] * 0.9
    dw = 1.6 * (dz2 > 0.15) * dfade * (0.6 + 0.4 * np.sin(DPH + 2 * np.pi * t / LOOP) ** 2)
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
    img += HAZE * 0.025 * np.exp(-((yy - hy) / (H * 0.04)) ** 2)      # only a whisper of horizon glow
    # per-point glow: every sheet dot gets its own halo, scaled by its brightness (bright dots glow most)
    gw_ = w * (0.25 + 0.75 * val) ** 1.5
    gacc = np.stack([splat(u / 2, v / 2, gw_ * col[:, c], H // 2, W // 2) for c in range(3)], -1)
    halo = cv2.GaussianBlur(gacc, (0, 0), 2.0) * 1.0 + cv2.GaussianBlur(gacc, (0, 0), 6.0) * 0.8
    img = img + cv2.resize(halo, (W, H)) * EXP * GLOW * (0.9 + 0.2 * pulse)
    img = img * vig
    L = luma(img)[..., None]
    Lt = L / (1 + L * 0.85)
    return np.clip(img * (Lt / np.maximum(L, 1e-5)) * 1.25, 0, 1)


if __name__ == '__main__':
    N = int(LOOP * FPS)                       # 450 frames; frame 450 == frame 0
    SPEED = 0.78                              # world units / s: a glide matched to the blob evolution
    taus, trav = [0.0], [0.0]
    for i in range(N):
        p = soft_pulse(i / FPS)
        taus.append(taus[-1] + (1.0 + 1.6 * p) / FPS)
        trav.append(trav[-1] + SPEED * (1.0 + 0.30 * p) / FPS)     # soft forward surge on each beat
    TAU_T = taus[N]
    D = round(trav[N] / DZW) * DZW                                 # whole number of dot rows per loop
    trav = [x * D / trav[N] for x in trav]
    TX = 4.0                                                       # last 4 s blend into the loop start
    wr = Writer('out/D_sheet_loop15.mp4', 'out/stills_loop', [0, 120, 240, 360, 449])
    for i in range(N):
        t = i / FPS
        p = soft_pulse(t)
        tr = trav[i]
        wz = np.floor(tr / DZW + 1e-6) * DZW
        frac = tr - wz
        n, fold = field_raw(taus[i], wz)
        k = np.clip((t - (LOOP - TX)) / TX, 0, 1)
        k = k * k * (3 - 2 * k)
        if k > 0:                                                  # variance-preserving blend toward t-15
            n2, fold2 = field_raw(taus[i] - TAU_T, wz - D)
            th = k * np.pi / 2
            n = n * np.cos(th) + n2 * np.sin(th)
            fold = fold * np.cos(th) + fold2 * np.sin(th)
        wr.write(render(shape(n, p), fold, frac, t, p))
    wr.close()
    print('travel per loop', D, 'tau', TAU_T)
