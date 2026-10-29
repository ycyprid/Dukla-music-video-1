"""3D halftone sheet v2, after the reference: dense even dot grid on one folding sheet, brightness from
fold density, very shallow depth of field with disc bokeh, floating dust, slow fluid camera.
Colour: our lifted palette, driven by the v2 field (same tempo surges, 100 BPM)."""
import sys
import opensimplex
from common import *

opensimplex.seed(70)
DUR = float(sys.argv[1]) if len(sys.argv) > 1 else 10.0
OUT = sys.argv[2] if len(sys.argv) > 2 else 'out/D_blobs_3d_v2.mp4'


def hexrgb(h):
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


PAL = gradient_lut([(0.00, hexrgb('5A0620')), (0.14, hexrgb('8A0A34')), (0.30, hexrgb('C2183F')),
                    (0.48, hexrgb('D8542C')), (0.64, hexrgb('E8A322')), (0.80, hexrgb('F6B93A')),
                    (0.92, hexrgb('FFDC96')), (1.00, hexrgb('FFF1D8'))])

NX, NZ = 420, 260
X0, X1, Z0, Z1 = -4.2, 4.2, 0.2, 9.0
gxw = np.linspace(X0, X1, NX).astype(np.float32)
gzw = np.linspace(Z0, Z1, NZ).astype(np.float32)
XW, ZW = np.meshgrid(gxw, gzw)
S = 0.8
RES = 0.03
UA, UB = X0 * S - 3, X1 * S + 3
VA, VB = Z0 * S - 3, Z1 * S + 6
uu = np.arange(UA, UB, RES); vv = np.arange(VA, VB, RES)


def lattice(z, sc=1.0):
    return opensimplex.noise3array(uu * sc, vv * sc, np.array([z]))[0].astype(np.float32)


def sample(lat, u, v):
    return cv2.remap(lat, ((u - UA) / RES).astype(np.float32), ((v - VA) / RES).astype(np.float32),
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def field(tau, pulse, flow):
    z = tau * 0.42
    u = XW * S; v = ZW * S + flow
    wx = sample(lattice(z * 0.6 + 10, 0.6), u, v)
    wy = sample(lattice(z * 0.6 + 40, 0.6), u, v)
    n = sample(lattice(z), u + 0.9 * wx + tau * 0.06, v + 0.9 * wy)
    lo = 0.02 - 0.06 * pulse
    val = np.clip((n - lo) / (0.62 - lo), 0, 1)
    val = val * val * (3 - 2 * val)
    # large folds of the sheet (the reference's silk waves), slow and fluid
    fold = sample(lattice(z * 0.30 + 70, 0.35), u, v)
    return val, fold


# dust: floating particles in the volume, mostly out of focus
rng = np.random.default_rng(11)
ND = 520
DX = rng.uniform(-3.5, 3.5, ND).astype(np.float32)
DY = rng.uniform(-0.2, 1.6, ND).astype(np.float32)
DZ = rng.uniform(0.25, 6.5, ND).astype(np.float32)
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
    d = np.clip(r + 0.5 - np.sqrt(xx * xx + yy * yy), 0, 1).astype(np.float32)
    d[np.sqrt(xx * xx + yy * yy) > r - 0.8] *= 1.25            # bright rim, like lens bokeh
    return d / d.sum()


# blur levels: small = gaussian dots (full res), large = disc bokeh (half res)
LEVELS = [('g', 0.7), ('g', 1.3), ('g', 2.2), ('d', 2.2), ('d', 3.6), ('d', 5.5), ('d', 8.0), ('d', 11.5)]
KER = {L: disc(r) for L, (k, r) in enumerate(LEVELS) if k == 'd'}
RAD = np.array([r if k == 'g' else r * 2 for k, r in LEVELS])   # in full-res pixels
vig = vignette(k=0.5)
EXP = 1.7
FPX = H * 1.05
ZF = 2.15
APER = 30.0


def project(xc, yc, zc, pitch):
    cp, sp = np.cos(pitch), np.sin(pitch)
    y2 = yc * cp + zc * sp; z2 = -yc * sp + zc * cp
    return W / 2 + FPX * xc / np.maximum(z2, 0.05), H / 2 - FPX * y2 / np.maximum(z2, 0.05), z2


def render(tau, t, pulse):
    flow = t * 0.06
    val, fold = field(tau, pulse, flow)
    Y = fold * 0.55 + val * 0.30                                    # sheet folds, bright areas rise further
    # slow fluid camera: gentle drift and bob
    camx = 0.30 * np.sin(t * 0.13)
    camy = 1.05 + 0.06 * np.sin(t * 0.21 + 1.0)
    pitch = 0.50 + 0.025 * np.sin(t * 0.17)          # look down so the sheet fills the frame
    u, v, z2 = project(XW - camx, Y - camy, ZW, pitch)
    u, v, z2, val = u.ravel(), v.ravel(), z2.ravel(), val.ravel()
    ok = z2 > 0.15
    col = PAL[((0.10 + 0.90 * val) * 255).astype(np.uint8)]
    farw = np.clip((z2 - 3.0) / 4.0, 0, 1)[:, None]                # distance shifts reds toward warm gold, not pink
    col = col * (1 - farw * 0.45) + PAL[np.full(col.shape[0], int(0.66 * 255))] * farw * 0.45
    w = (0.30 + 0.70 * val) * ok * 2.0 / (1 + (z2 / 2.4) ** 2.2)          # compensate far dots piling up on screen
    coc = APER * np.abs(1 / np.maximum(z2, 0.15) - 1 / ZF)
    size = FPX * 0.004 / np.maximum(z2, 0.15)
    r = np.maximum(size, coc)
    # dust particles: slow float, always some bokeh
    dx = DX + 0.05 * np.sin(t * 0.4 + DPH) - camx
    dy = DY + 0.06 * np.sin(t * 0.3 + DPH * 1.3) + 0.02 * t - camy
    du, dv, dz = project(dx, dy, DZ, pitch)
    dcoc = APER * np.abs(1 / np.maximum(dz, 0.15) - 1 / ZF)
    dok = dz > 0.15
    dcol = PAL[np.full(ND, int(0.78 * 255))] * 0.9
    dw = 3.0 * dok * (0.6 + 0.4 * np.sin(DPH + t * 0.5) ** 2)
    U = np.concatenate([u, du]); V = np.concatenate([v, dv])
    dcoc = dcoc * 1.6
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
            b = cv2.filter2D(acc, -1, KER[L]) * (rad / 2.2) ** 1.2 * 1.6
            img += cv2.resize(b, (W, H), interpolation=cv2.INTER_LINEAR)
    img *= EXP
    # per-dot glow (bright dots radiate)
    gsm = cv2.GaussianBlur(cv2.resize(img, (W // 4, H // 4), interpolation=cv2.INTER_AREA), (0, 0), 3)
    img = img + cv2.resize(np.clip(gsm - 0.25, 0, None), (W, H)) * (0.6 + 0.2 * pulse)
    img = img * vig
    L = luma(img)[..., None]
    Lt = L / (1 + L * 0.85)                                         # hue-preserving roll-off
    return np.clip(img * (Lt / np.maximum(L, 1e-5)) * 1.25, 0, 1)


if __name__ == '__main__':
    wr = Writer(OUT, 'out/stills_D3v2', [30, 150, 270])
    tau = 0.0
    for i in range(int(DUR * FPS)):
        t = i / FPS
        p = soft_pulse(t)
        tau += (1.0 + 1.6 * p) / FPS
        wr.write(render(tau, t, p))
    wr.close()
