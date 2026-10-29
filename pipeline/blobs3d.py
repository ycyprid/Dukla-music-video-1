"""Halftone blobs in 3D: the v2 field (same motion, same tempo surges, same palette) drives a sheet of dots.
Each dot rises by its brightness; the sheet itself undulates gently; low grazing camera, depth of field,
per-dot glow by brightness. 100 BPM grid shared with test A."""
import sys
import opensimplex
from common import *

opensimplex.seed(70)
DUR = float(sys.argv[1]) if len(sys.argv) > 1 else 10.0


def hexrgb(h):
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


PAL = gradient_lut([(0.00, hexrgb('3C0014')), (0.14, hexrgb('8A0A34')), (0.30, hexrgb('C2183F')),
                    (0.48, hexrgb('D8542C')), (0.64, hexrgb('E8A322')), (0.80, hexrgb('F6B93A')),
                    (0.92, hexrgb('FFDC96')), (1.00, hexrgb('FFF1D8'))])

# dot sheet in world space (X across, Z away from camera)
NX, NZ = 260, 170
X0, X1, Z0, Z1 = -3.4, 3.4, 0.5, 6.5
gxw = np.linspace(X0, X1, NX).astype(np.float32)
gzw = np.linspace(Z0, Z1, NZ).astype(np.float32)
XW, ZW = np.meshgrid(gxw, gzw)
S = 1.05                                   # noise units per world unit (blob size ~ v2)
rng = np.random.default_rng(3)
JIT = rng.uniform(-0.25, 0.25, XW.shape).astype(np.float32)   # tiny jitter so rows don't moire

# regular noise lattice that warped lookups sample from (fast, vectorised)
RES = 0.03
UA, UB = X0 * S - 3, X1 * S + 3
VA, VB = Z0 * S - 3, Z1 * S + 6
uu = np.arange(UA, UB, RES); vv = np.arange(VA, VB, RES)


def lattice(z, off=0.0, sc=1.0):
    return opensimplex.noise3array(uu * sc + off, vv * sc + off, np.array([z]))[0].astype(np.float32)


def sample(lat, u, v):
    return cv2.remap(lat, ((u - UA) / RES).astype(np.float32), ((v - VA) / RES).astype(np.float32),
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def field(tau, pulse, flow):
    z = tau * 0.42
    u = XW * S; v = ZW * S + flow
    wx = sample(lattice(z * 0.6 + 10, 0, 0.6), u, v)
    wy = sample(lattice(z * 0.6 + 40, 0, 0.6), u, v)
    n = sample(lattice(z), u + 0.9 * wx + tau * 0.06, v + 0.9 * wy)
    lo = 0.02 - 0.06 * pulse
    val = np.clip((n - lo) / (0.62 - lo), 0, 1)
    val = val * val * (3 - 2 * val)
    swell = sample(lattice(z * 0.35 + 70, 0, 0.3), u, v)            # slow undulation of the whole sheet
    return val, swell


def splat(u, v, w):
    img = np.zeros(H * W, np.float32)
    x0 = np.floor(u).astype(np.int64); y0 = np.floor(v).astype(np.int64)
    fx = (u - x0).astype(np.float32); fy = (v - y0).astype(np.float32)
    for dx, dy, ww in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)), (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
        xi, yi = x0 + dx, y0 + dy
        ok = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
        img += np.bincount(yi[ok] * W + xi[ok], weights=(ww * w)[ok], minlength=H * W)
    return img.reshape(H, W)


def splat_rgb(u, v, col, w):
    return np.stack([splat(u, v, w * col[:, c]) for c in range(3)], -1)


SIGMAS = [0.8, 1.5, 2.8, 5.0, 8.5]
vig = vignette(k=0.45)
FPX = H * 1.15
PITCH = 0.30                                # look down ~17 degrees
cp, sp = np.cos(PITCH), np.sin(PITCH)
ZF = 2.3                                    # focus distance


def render(tau, t, pulse):
    flow = t * 0.10                                                   # surface glides slowly toward camera
    val, swell = field(tau, pulse, flow)
    Y = val * 0.42 + swell * 0.22                                     # dots rise by brightness, sheet undulates
    camx = 0.35 * np.sin(t * 0.11); camy = 1.0
    xc = (XW + JIT * (X1 - X0) / NX) - camx
    yc = Y - camy
    zc = ZW
    # camera tilted down by PITCH: forward = (0,-sin,cos), up = (0,cos,sin)
    y2 = yc * cp + zc * sp
    z2 = -yc * sp + zc * cp
    ok = z2 > 0.2
    u = W / 2 + FPX * xc / z2
    vs = H / 2 - FPX * y2 / z2
    vf = val.ravel(); uf = u.ravel(); vf_s = vs.ravel(); z2f = z2.ravel(); okf = ok.ravel()
    col = PAL[((0.16 + 0.84 * np.clip(vf, 0, 1)) * 255).astype(np.uint8)]   # floor dots stay visible burgundy
    bright = (0.24 + 0.76 * vf ** 1.15) * np.exp(-np.maximum(z2f - 1.2, 0) / 3.8) * okf
    size = FPX * (0.006 + 0.009 * np.sqrt(vf)) / np.maximum(z2f, 0.2)
    coc = 11.0 * np.abs(1 / np.maximum(z2f, 0.2) - 1 / ZF)
    sig = np.maximum(size * 0.55, coc)
    layer = np.argmin(np.abs(np.array(SIGMAS)[:, None] - sig[None]), 0)
    img = np.zeros((H, W, 3), np.float32)
    for L, sg in enumerate(SIGMAS):
        m = layer == L
        if not m.any():
            continue
        comp = (sg / 0.8) ** 1.45
        acc = splat_rgb(uf[m], vf_s[m], col[m], bright[m] * comp)
        img += cv2.GaussianBlur(acc, (0, 0), sg)
    # per-dot glow: bright dots radiate by their own brightness
    gw = bright * vf ** 2.2
    gacc = splat_rgb(uf, vf_s, col, gw)
    small = cv2.resize(gacc, (W // 2, H // 2), interpolation=cv2.INTER_AREA) * 4
    halo = cv2.GaussianBlur(small, (0, 0), 3.0) * 0.9 + cv2.GaussianBlur(small, (0, 0), 10) * 0.6
    img = img * 2.6 + cv2.resize(halo, (W, H)) * (0.9 + 0.25 * pulse)
    return np.clip(tonemap(img * vig), 0, 1)


if __name__ == '__main__':
    wr = Writer('out/D_blobs_3d.mp4', 'out/stills_D3', [30, 150, 270])
    tau = 0.0
    for i in range(int(DUR * FPS)):
        t = i / FPS
        p = soft_pulse(t)
        tau += (1.0 + 1.6 * p) / FPS
        wr.write(render(tau, t, p))
    wr.close()
