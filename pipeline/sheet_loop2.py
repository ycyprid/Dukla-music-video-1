"""3D dot sheet, 15 s seamless loop, v2.
Fixes/additions vs loop v1:
 - no seams: each dot is split smoothly between the two nearest focus levels (no snapping)
 - no exposed edge: sheet starts right under the camera and is wider at the sides
 - steadier brightness: pass 1 measures each frame's brightness from the dots themselves,
   pass 2 applies a smoothed, loop-wrapped correction (partial, so the landscape still breathes)
 - subtle twinkle: individual ground-anchored dots glint; periodic per loop (seamless)
 - dust: ~1/3 as many specks, much smaller and fainter
Look, palette, tempo, horizon (1/4), travel and loop logic as loop v1."""
import sys
import opensimplex
from common import *

opensimplex.seed(70)


def hexrgb(h):
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


PAL = gradient_lut([(0.00, hexrgb('5A0620')), (0.14, hexrgb('8A0A34')), (0.30, hexrgb('C2183F')),
                    (0.48, hexrgb('D8542C')), (0.64, hexrgb('E8A322')), (0.80, hexrgb('F6B93A')),
                    (0.92, hexrgb('FFDC96')), (1.00, hexrgb('FFF1D8'))])
HAZE = np.array(hexrgb('E39A3C')[::-1], np.float32) / 255
DEEP = np.array(hexrgb('4A0418')[::-1], np.float32) / 255

FPX = H * 1.05
CAMY = 1.05
ZNEAR, ZFAR = 0.12, 20.0                       # sheet now starts under the camera (no exposed near edge)
DXW, DZW = 0.032, 0.040
S = 0.8
RES = 0.035
HORIZON = 0.25
LOOP = 15.0
SPEED = 0.78

rows = np.arange(ZNEAR, ZFAR, DZW, dtype=np.float32)
pts_x, pts_z, pts_c = [], [], []
for z in rows:
    half = max(z, 0.5) * (W / 2) / FPX * 1.3 + 1.4    # wider margin: camera sway never reveals the sides
    xs = np.arange(-half, half, DXW, dtype=np.float32)
    pts_x.append(xs); pts_z.append(np.full_like(xs, z))
PX = np.concatenate(pts_x); PZ = np.concatenate(pts_z)
COL_ID = np.round(PX / DXW).astype(np.int64)
UA, UB = PX.min() * S - 3, PX.max() * S + 3
VA, VB = PZ.min() * S - 3, PZ.max() * S + 8
uu = np.arange(UA, UB, RES); vv = np.arange(VA, VB, RES)


def lattice(z, sc=1.0):
    return opensimplex.noise3array(uu * sc, vv * sc, np.array([z]))[0].astype(np.float32)


def sample(lat, u, v):
    n = u.size; pad = (-n) % 1024
    mu = np.pad(((u - UA) / RES).astype(np.float32), (0, pad)).reshape(-1, 1024)
    mv = np.pad(((v - VA) / RES).astype(np.float32), (0, pad)).reshape(-1, 1024)
    return cv2.remap(lat, mu, mv, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT).ravel()[:n]


def field_raw(tau, wz):
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


# ---- timing (identical to loop v1) ----
N = int(LOOP * FPS)
taus, trav = [0.0], [0.0]
for _i in range(N):
    _p = soft_pulse(_i / FPS)
    taus.append(taus[-1] + (1.0 + 1.6 * _p) / FPS)
    trav.append(trav[-1] + SPEED * (1.0 + 0.30 * _p) / FPS)
TAU_T = taus[N]
D = round(trav[N] / DZW) * DZW
ROWS_PER_LOOP = int(round(D / DZW))
trav = [x * D / trav[N] for x in trav]
TX = 4.0


def frame_field(i):
    t = i / FPS; p = soft_pulse(t); tr = trav[i]
    wz = np.floor(tr / DZW + 1e-6) * DZW
    n, fold = field_raw(taus[i], wz)
    k = np.clip((t - (LOOP - TX)) / TX, 0, 1); k = k * k * (3 - 2 * k)
    if k > 0:
        n2, f2 = field_raw(taus[i] - TAU_T, wz - D); th = k * np.pi / 2
        n = n * np.cos(th) + n2 * np.sin(th); fold = fold * np.cos(th) + f2 * np.sin(th)
    row_id = np.round((PZ + wz) / DZW).astype(np.int64) % ROWS_PER_LOOP   # periodic world rows -> seamless
    return shape(n, p), fold, tr - wz, t, p, row_id


# ---- twinkle: per ground dot, integer cycles per loop ----
def hash01(a, b, salt):
    h = (a * 73856093) ^ (b * 19349663) ^ salt
    h = (h ^ (h >> 13)) * 1274126177
    return ((h ^ (h >> 16)) & 0xFFFF).astype(np.float32) / 65535.0


def twinkle(row_id, t):
    ph = hash01(row_id, COL_ID, 911)
    k = 1 + np.floor(hash01(row_id, COL_ID, 4242) * 3)              # 1..3 glints per loop
    s = np.sin(2 * np.pi * (k * t / LOOP + ph))
    return np.maximum(s, 0) ** 24                                    # short, sparse glints


# ---- dust: fewer, finer ----
rng = np.random.default_rng(11)
ND = 110
DX = rng.uniform(-3.5, 3.5, ND).astype(np.float32)
DY = rng.uniform(-0.3, 1.2, ND).astype(np.float32)
DZMIN, DZRANGE = 0.3, 7.5
DZ0 = rng.uniform(0, DZRANGE, ND).astype(np.float32)
DPH = rng.uniform(0, 6.28, ND).astype(np.float32)
DK = rng.integers(1, 3, ND).astype(np.float32)


def dust_z(t):
    zz = np.mod(DZ0 - DZRANGE * t / LOOP, DZRANGE)
    fade = np.clip(zz / 0.6, 0, 1) * np.clip((DZRANGE - zz) / 1.2, 0, 1)
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
LRAD = np.log(RAD)
vig = vignette(k=0.45)
APER = 10.0
NEAR_GAIN = 9.0
GLOW = 1.0
EXP = 1.7
PITCH = float(np.arctan((0.5 - HORIZON) * H / FPX) + np.arctan(CAMY / 15.5))
CP, SP = np.cos(PITCH), np.sin(PITCH)
ZF = CAMY / np.tan(PITCH + 0.12) * 1.6


def geometry(val, fold, frac, t, row_id):
    Y = fold * 0.55 + val * 0.30
    camx = 0.30 * np.sin(2 * np.pi * t / LOOP)
    yc = Y - CAMY; xc = PX - camx; zc = PZ - frac
    y2 = yc * CP + zc * SP; z2 = -yc * SP + zc * CP
    ok = z2 > 0.15
    z2c = np.maximum(z2, 0.15)
    u = W / 2 + FPX * xc / z2c; v = H / 2 - FPX * y2 / z2c
    col = PAL[((0.10 + 0.90 * val) * 255).astype(np.uint8)]
    fog = np.clip((z2 - 2.5) / 12.0, 0, 1)[:, None] ** 0.8
    col = col * (1 - fog * 0.7) + DEEP * fog * 0.7
    w = (0.30 + 0.70 * val) * ok * 2.0 * np.minimum(1.0, (1.3 / z2c) ** 2) \
        * np.exp(-np.maximum(z2 - 1.3, 0) / 6.5) * NEAR_GAIN
    w = w * (1 - np.clip((z2 - 15.0) / 5.0, 0, 1))
    tw = twinkle(row_id, t) * (0.4 + 0.6 * val)                    # glints show most on the bright folds
    r = np.maximum(FPX * 0.004 / z2c, APER * np.abs(1 / z2c - 1 / ZF))
    inframe = (u > -40) & (u < W + 40) & (v > -40) & (v < H + 40) & ok
    return u, v, col, w, tw, r, inframe, camx


def brightness_proxy(i):
    val, fold, frac, t, p, row_id = frame_field(i)
    u, v, col, w, tw, r, inf, _ = geometry(val, fold, frac, t, row_id)
    return float(np.sum((w * (1 + 1.5 * tw) * luma(col))[inf]))


def render(i, gain):
    val, fold, frac, t, p, row_id = frame_field(i)
    u, v, col, w, tw, r, inf, camx = geometry(val, fold, frac, t, row_id)
    w = w * gain * (1 + 1.5 * tw)
    # dust: fewer, finer specks streaming toward camera
    DZd, dfade = dust_z(t)
    ph = 2 * np.pi * DK * t / LOOP
    dx = DX + 0.05 * np.sin(ph + DPH) - camx
    dy = DY + 0.06 * np.sin(ph + DPH * 1.3) - CAMY
    dz2 = -dy * SP + DZd * CP; dy2 = dy * CP + DZd * SP
    dz2c = np.maximum(dz2, 0.15)
    du = W / 2 + FPX * dx / dz2c; dv = H / 2 - FPX * dy2 / dz2c
    dr = np.maximum(30.0 * np.abs(1 / dz2c - 1 / ZF) * 0.55, 1.0)
    dcol = PAL[np.full(ND, int(0.80 * 255))]
    dw = 0.55 * (dz2 > 0.15) * dfade * (0.6 + 0.4 * np.sin(DPH + 2 * np.pi * t / LOOP) ** 2)
    U = np.concatenate([u[inf], du]); V = np.concatenate([v[inf], dv])
    C = np.concatenate([col[inf], dcol]); Wt = np.concatenate([w[inf], dw]); Rr = np.concatenate([r[inf], dr])
    # smooth focus: split each dot between the two nearest blur levels (log-radius interpolation) -> no seams
    lr = np.clip(np.log(Rr), LRAD[0], LRAD[-1])
    k0 = np.clip(np.searchsorted(LRAD, lr) - 1, 0, len(RAD) - 2)
    f = ((lr - LRAD[k0]) / (LRAD[k0 + 1] - LRAD[k0])).astype(np.float32)
    img = np.zeros((H, W, 3), np.float32)
    for L, (kind, rad) in enumerate(LEVELS):
        wl = np.where(k0 == L, 1 - f, 0) + np.where(k0 + 1 == L, f, 0)
        m = wl > 1e-4
        if not m.any():
            continue
        ww = Wt[m] * wl[m]
        if kind == 'g':
            acc = np.stack([splat(U[m], V[m], ww * C[m, c], H, W) for c in range(3)], -1)
            img += cv2.GaussianBlur(acc, (0, 0), rad) * (rad / 0.7) ** 1.2
        else:
            acc = np.stack([splat(U[m] / 2, V[m] / 2, ww * C[m, c], H // 2, W // 2) for c in range(3)], -1)
            img += cv2.resize(cv2.filter2D(acc, -1, KER[L]) * (rad / 2.2) ** 1.2 * 1.6, (W, H))
    img *= EXP
    hy = H / 2 - FPX * np.tan(PITCH - np.arctan(CAMY / 15.5))
    yy = np.arange(H, dtype=np.float32)[:, None, None]
    img += HAZE * 0.025 * np.exp(-((yy - hy) / (H * 0.04)) ** 2)
    # per-point glow (twinkling dots flare a little more)
    gw_ = w[inf] * (0.25 + 0.75 * val[inf]) ** 1.5 * (1 + 1.5 * tw[inf])
    ui, vi, ci = u[inf], v[inf], col[inf]
    gacc = np.stack([splat(ui / 2, vi / 2, gw_ * ci[:, c], H // 2, W // 2) for c in range(3)], -1)
    halo = cv2.GaussianBlur(gacc, (0, 0), 2.0) * 1.0 + cv2.GaussianBlur(gacc, (0, 0), 6.0) * 0.8
    img = img + cv2.resize(halo, (W, H)) * EXP * GLOW * (0.9 + 0.2 * p)
    img = img * vig
    Lm = luma(img)[..., None]
    Lt = Lm / (1 + Lm * 0.85)
    return np.clip(img * (Lt / np.maximum(Lm, 1e-5)) * 1.25, 0, 1)


if __name__ == '__main__':
    # pass 1: brightness of every frame, from the dots themselves (no image render)
    B = np.array([brightness_proxy(i) for i in range(N)])
    # loop-wrapped smoothing (~1.5 s) so the correction is gentle and the seam stays seamless
    k = np.exp(-0.5 * (np.arange(-60, 61) / 22.0) ** 2); k /= k.sum()
    Bs = np.real(np.fft.ifft(np.fft.fft(B) * np.fft.fft(np.roll(np.pad(k, (0, N - k.size)), -60))))
    target = np.median(Bs)
    gain = np.clip((target / Bs) ** 0.65, 0.75, 1.8)               # partial: evens dips, keeps some breathing
    np.save('out/loop2_gain.npy', np.stack([B, Bs, gain]))
    import os
    if os.environ.get('GAINFILE'):                                   # refined gain measured from a previous render
        gain = np.load(os.environ['GAINFILE'])
    print('brightness range before %.2f-%.2f  gain %.2f-%.2f' % (Bs.min() / target, Bs.max() / target, gain.min(), gain.max()), flush=True)
    wr = Writer(os.environ.get('OUTFILE', 'out/D_sheet_loop15_v2.mp4'), 'out/stills_loop2', [0, 45, 120, 225, 330, 420, 449])
    for i in range(N):
        wr.write(render(i, gain[i]))
    wr.close()
