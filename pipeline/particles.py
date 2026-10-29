"""Depth-driven particle renders in the club palette.
usage: python3 particles.py leaves|roses|sunflowers [seconds]"""
import sys, glob
from common import *

MODE = sys.argv[1]
DUR = float(sys.argv[2]) if len(sys.argv) > 2 else 10
SRC = {'leaves': 'src/h5.mp4', 'roses': 'src/h6.mp4', 'sunflowers': 'src/h7.mp4'}[MODE]
DEP = f'depth/{MODE}'


def hexrgb(h):
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


# PDF direction, pushed brighter / more saturated
PAL = gradient_lut([(0.00, hexrgb('1E000A')), (0.20, hexrgb('7A0A2C')), (0.38, hexrgb('B0123F')),
                    (0.58, hexrgb('D8892A')), (0.78, hexrgb('F2C146')), (1.00, hexrgb('FFF4D6'))])

rng = np.random.default_rng(70)


def splat(u, v, col, w, img=None):
    """bilinear additive splat of N points into HxWx3"""
    if img is None:
        img = np.zeros((H * W, 3), np.float32)
    x0 = np.floor(u).astype(np.int64); y0 = np.floor(v).astype(np.int64)
    fx = (u - x0).astype(np.float32); fy = (v - y0).astype(np.float32)
    for dx, dy, ww in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)), (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
        xi, yi = x0 + dx, y0 + dy
        ok = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
        idx = yi[ok] * W + xi[ok]
        wt = (ww * w)[ok]
        for c in range(3):
            img[:, c] += np.bincount(idx, weights=wt * col[ok, c], minlength=H * W).astype(np.float32)
    return img


def finish(acc, pulse, glow=1.0):
    img = acc.reshape(H, W, 3) * EXPO
    soft = cv2.GaussianBlur(img, (0, 0), 1.6)
    img = img * 0.75 + soft * 0.9                         # crisp points + soft body
    img = bloom(img, 0.55, (0.45 + 0.12 * pulse) * glow, 12)
    img = img * vig
    return grain(tonemap(img), 0.01)


vig = vignette(k=0.5)
src = Clip(SRC, int(DUR * FPS) + 2)
dfiles = sorted(glob.glob(f'{DEP}/d*.png'))
N = min(int(DUR * FPS), len(dfiles), src.n)

# particle grid (persistent identities) ------------------------------------------------
STEP = {'leaves': 3, 'roses': 3, 'sunflowers': 2}[MODE]
gy, gx = np.mgrid[0:H:STEP, 0:W:STEP]
gx = gx.ravel().astype(np.float32); gy = gy.ravel().astype(np.float32)
jx = rng.uniform(-0.5, 0.5, gx.size).astype(np.float32) * STEP
jy = rng.uniform(-0.5, 0.5, gx.size).astype(np.float32) * STEP
seed = rng.random(gx.size).astype(np.float32)
if MODE == 'sunflowers':        # scan lines: keep every 6th row of particles
    keep = (gy.astype(int) % 6) == 0
    gx, gy, jx, jy, seed = gx[keep], gy[keep], jx[keep] * 0.3, jy[keep] * 0, seed[keep]
sx = np.clip(gx + jx, 0, W - 1); sy = np.clip(gy + jy, 0, H - 1)
ix, iy = sx.astype(np.int32), sy.astype(np.int32)
EXPO = {'leaves': 9.0, 'roses': 9.0, 'sunflowers': 3.2}[MODE] * 0.9

# smooth noise fields for drift (low-res random, upscaled, two time keys)
def noise_field(k):
    r = np.random.default_rng(k)
    a = r.standard_normal((9, 16)).astype(np.float32)
    return cv2.resize(a, (W, H), interpolation=cv2.INTER_CUBIC)
NF = [(noise_field(10 + k), noise_field(50 + k)) for k in range(6)]

def drift(t, speed=0.25):
    s = t * speed
    k = int(s) % 5; f = s - int(s); f = f * f * (3 - 2 * f)
    nx = NF[k][0] * (1 - f) + NF[k + 1][0] * f
    ny = NF[k][1] * (1 - f) + NF[k + 1][1] * f
    return nx[iy, ix], ny[iy, ix]


def project(X, Y, Z, yaw, pitch, cam=2.2, fpx=None):
    """world coords (units of screen height, Z forward) -> pixels"""
    fpx = fpx or H * cam * 1.0
    cy, sy_ = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    X1 = X * cy + Z * sy_; Z1 = -X * sy_ + Z * cy
    Y1 = Y * cp - Z1 * sp; Z2 = Y * sp + Z1 * cp
    zc = Z2 + cam
    return W / 2 + fpx * X1 / zc, H / 2 + fpx * Y1 / zc, zc


wr = Writer(f'out/P_{MODE}.mp4', f'out/stills_P_{MODE}', [60, 150, 255])
X0 = (sx - W / 2) / H; Y0 = (sy - H / 2) / H
_e = np.minimum(np.minimum(sx, W - 1 - sx) / 160, np.minimum(sy, H - 1 - sy) / 110)
EDGE = np.clip(_e, 0, 1) ** 1.5   # particles dissolve toward the frame edge: no hard cloud border
for i in range(N):
    t = i / FPS
    pulse = soft_pulse(t)
    f = src.get(i).astype(np.float32) / 255
    d = cv2.imread(dfiles[i], -1).astype(np.float32) / 65535
    dp = d[iy, ix]                                   # 1 = near
    c = f[iy, ix]
    lum = luma(c)

    if MODE == 'leaves':
        # constellation: leaves float in black space, sky removed, slow orbit
        sky = (c[:, 0] > c[:, 2] * 1.05) | (dp < 0.10)          # blue-ish or far
        Z = (1 - dp) * (0.9 + 0.08 * pulse)
        yaw = 0.16 * np.sin(t * 0.25); pitch = 0.05 * np.sin(t * 0.17)
        X, Y = X0.copy(), Y0.copy()
        # a few motes break free and rise
        mote = seed < 0.06
        ph = (t * 0.12 + seed * 7) % 1.0
        nx, ny = drift(t)
        X[mote] += nx[mote] * 0.05 * ph[mote]
        Y[mote] -= ph[mote] * 0.35
        Z[mote] -= ph[mote] * 0.2
        u, v, zc = project(X, Y, Z - 0.45, yaw, pitch)
        idx = np.clip(lum * 0.7 + dp * 0.45, 0, 1)
        w = (0.25 + 1.1 * lum) * (0.35 + 0.65 * dp) * (~sky)
        w[mote] *= np.sin(ph[mote] * np.pi) * 1.6
        w = w * (2.2 / zc) ** 2 * 0.55
        w = w * EDGE
        col = PAL[(idx * 255).astype(np.uint8)]
        wr.write(finish(splat(u, v, col, w), pulse))

    elif MODE == 'roses':
        # petal dust: the bouquet breathes apart into drifting gold dust and gathers again
        a = 0.5 - 0.5 * np.cos(2 * np.pi * t / DUR)             # 0 -> 1 -> 0
        a = a ** 1.6
        nx, ny = drift(t, 0.18)
        spread = a * (0.25 + 0.75 * seed)                        # particles leave at different times
        X = X0 + nx * 0.09 * spread
        Y = Y0 + ny * 0.06 * spread - spread * 0.22 * (0.3 + seed)
        Z = (1 - dp) * 0.8 + spread * (seed - 0.5) * 0.6
        yaw = 0.06 * np.sin(t * 0.2)
        u, v, zc = project(X, Y, Z - 0.4, yaw, 0.0, cam=2.2 - 0.12 * t / DUR)
        dark = lum < 0.06
        idx = np.clip(lum * 0.85 + dp * 0.35 + spread * 0.25, 0, 1)
        w = (0.2 + 1.3 * lum) * (0.4 + 0.6 * dp) * (~dark) * (1 - 0.45 * spread)
        w = w * (2.2 / zc) ** 2 * 0.6
        w = w * EDGE
        col = PAL[(idx * 255).astype(np.uint8)]
        wr.write(finish(splat(u, v, col, w), pulse))

    else:
        # depth scan: horizontal lines lifted by depth into a ridge landscape, a light sweep travels through depth
        Z = (1 - dp) * 1.1
        Y = Y0 - dp * 0.22 * (0.92 + 0.08 * pulse)               # nearer = lifted
        yaw = 0.10 * np.sin(t * 0.22); pitch = 0.22 + 0.04 * np.sin(t * 0.15)
        u, v, zc = project(X0, Y, Z - 0.55, yaw, pitch, cam=2.0)
        scan = (t * 0.22) % 1.4 - 0.2                              # sweep from near to far, repeating
        band = np.exp(-((dp - (1 - scan)) / 0.035) ** 2)
        idx = np.clip(lum * 0.6 + dp * 0.35 + band * 0.5, 0, 1)
        w = (0.3 + 1.0 * lum) * (0.45 + 0.55 * dp) * (1 + 2.5 * band)
        w = w * (2.0 / zc) ** 2 * 0.75
        w = w * EDGE
        col = PAL[(idx * 255).astype(np.uint8)]
        wr.write(finish(splat(u, v, col, w), pulse))
wr.close()
