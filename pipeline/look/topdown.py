"""Top-down version of the benchmark dot sheet (same field, palette, glow, twinkle, smooth focus, dust).
Camera looks down at the sheet (pitch 90 = straight down, ~72 = slight tilt); blobs rise toward camera.
usage: python3 topdown.py still <pitch_deg> <out.png> [t]"""
import sys
import numpy as np
import common
common.FPS = 25
import sheet_loop2 as L
import sheet_loop3 as M                      # 25 fps timing (tau / travel / pulses) of the benchmark loop
from common import W, H, luma

cv2 = L.cv2
CAMH = 4.2                                   # camera height above the floor of the sheet
DX_, DZ_ = 0.030, 0.030


def build_grid(pitch):
    reach = CAMH / np.tan(max(pitch - 0.445 - 0.12, 0.25))     # far edge of view on the floor (+ margin)
    near = CAMH / np.tan(min(pitch + 0.445 + 0.12, np.pi - 0.25))
    z0, z1 = min(near, -1.0) - 0.6, reach + 1.2
    half = max(abs(z0), abs(z1), CAMH) * 1.1 + 2.5
    xs = np.arange(-half, half, DX_, dtype=np.float32)
    zs = np.arange(z0, z1, DZ_, dtype=np.float32)
    X, Z = np.meshgrid(xs, zs)
    return X.ravel(), Z.ravel()


def field(x_w, z_w, tau, p):
    z = tau * 0.42
    u = x_w * L.S; v = z_w * L.S
    wx = L.sample(L.lattice(z * 0.6 + 10, 0.6), u, v)
    wy = L.sample(L.lattice(z * 0.6 + 40, 0.6), u, v)
    n = L.sample(L.lattice(z), u + 0.9 * wx + tau * 0.06, v + 0.9 * wy)
    fold = L.sample(L.lattice(z * 0.30 + 70, 0.35), u, v)
    return L.shape(n, p), L.shape(n, p * 0.3), fold


def twinkle(rid, cid, t):
    ph = L.hash01(rid, cid, 911)
    k = 1 + np.floor(L.hash01(rid, cid, 4242) * 3)
    return np.maximum(np.sin(2 * np.pi * (k * t / L.LOOP + ph)), 0) ** 24


rng = np.random.default_rng(5)
ND = 90
DUX = rng.uniform(-5, 5, ND).astype(np.float32); DUZ = rng.uniform(-3, 7, ND).astype(np.float32)
DUY = rng.uniform(1.4, 3.6, ND).astype(np.float32)            # floating between sheet and camera


def render(pitch_deg, i, gain=1.0, drift=(0.0, 0.0)):
    pitch = np.radians(pitch_deg)
    cp, sp = np.cos(pitch), np.sin(pitch)
    t = i / M.FPS; p = M.PULSES[i]; tau = M.taus[i]
    PXg, PZg = build_grid(pitch)
    # camera glides across the surface (world offset); grid snapped to the dot spacing -> ground-anchored dots
    ox, oz = drift
    sx, sz = np.floor(ox / DX_) * DX_, np.floor(oz / DZ_) * DZ_
    xw, zw = PXg + sx, PZg + sz
    val, valg, fold = field(xw, zw, tau, p)
    Y = fold * 0.45 + valg * 0.95                                   # taller domes: relief reads from above
    xc = PXg - (ox - sx); zc = PZg - (oz - sz); yc = Y - CAMH
    y2 = yc * cp + zc * sp; z2 = -yc * sp + zc * cp
    ok = z2 > 0.2
    z2c = np.maximum(z2, 0.2)
    FPX = L.FPX
    u = W / 2 + FPX * xc / z2c; v = H / 2 - FPX * y2 / z2c
    inf = ok & (u > -60) & (u < W + 60) & (v > -60) & (v < H + 60)
    col = L.PAL[((0.10 + 0.90 * val) * 255).astype(np.uint8)]
    zref = CAMH - 1.15                                            # distance of the highest dome tops
    zfloor = CAMH
    # closer = brighter (domes rising toward camera glow most); floor recedes into burgundy
    fog = np.clip((z2 - zref) / (zfloor + 1.5 - zref), 0, 1)[:, None] ** 0.9
    col = col * (1 - fog * 0.35) + L.DEEP * fog * 0.35
    w = (0.38 + 0.62 * val) * ok * np.minimum(1.0, (zref / z2c) ** 1.2) * np.exp(-np.maximum(z2 - zref, 0) / 9.0)
    w = w * 2.0 * GAIN_TD * gain
    rid = np.round(zw / DZ_).astype(np.int64); cid = np.round(xw / DX_).astype(np.int64)
    tw = twinkle(rid, cid, t) * (0.4 + 0.6 * val)
    w = w * (1 + 1.5 * tw)
    ZF = zref + 0.25                                              # focus on the dome tops
    r = np.maximum(FPX * 0.004 / z2c, APER_TD * np.abs(1 / z2c - 1 / ZF))
    U, V, C, Wt, Rr = u[inf], v[inf], col[inf], w[inf], r[inf]
    # dust floating between camera and sheet
    dyc = DUY - CAMH; dzc = DUZ - oz * 0.0
    dy2 = dyc * cp + dzc * sp; dz2 = -dyc * sp + dzc * cp
    dz2c = np.maximum(dz2, 0.2)
    du = W / 2 + FPX * (DUX - ox * 0.15) / dz2c; dv = H / 2 - FPX * dy2 / dz2c
    dr = np.maximum(30.0 * np.abs(1 / dz2c - 1 / ZF) * 0.55, 1.0)
    dcol = L.PAL[np.full(ND, int(0.80 * 255))]
    dw = 0.55 * (dz2 > 0.2)
    U = np.concatenate([U, du]); V = np.concatenate([V, dv]); C = np.concatenate([C, dcol])
    Wt = np.concatenate([Wt, dw]); Rr = np.concatenate([Rr, dr])
    lr = np.clip(np.log(Rr), L.LRAD[0], L.LRAD[-1])
    k0 = np.clip(np.searchsorted(L.LRAD, lr) - 1, 0, len(L.RAD) - 2)
    f = ((lr - L.LRAD[k0]) / (L.LRAD[k0 + 1] - L.LRAD[k0])).astype(np.float32)
    img = np.zeros((H, W, 3), np.float32)
    for Lv, (kind, rad) in enumerate(L.LEVELS):
        wl = np.where(k0 == Lv, 1 - f, 0) + np.where(k0 + 1 == Lv, f, 0)
        m = wl > 1e-4
        if not m.any():
            continue
        ww = Wt[m] * wl[m]
        if kind == 'g':
            acc = np.stack([L.splat(U[m], V[m], ww * C[m, c], H, W) for c in range(3)], -1)
            img += cv2.GaussianBlur(acc, (0, 0), rad) * (rad / 0.7) ** 1.2
        else:
            acc = np.stack([L.splat(U[m] / 2, V[m] / 2, ww * C[m, c], H // 2, W // 2) for c in range(3)], -1)
            img += cv2.resize(cv2.filter2D(acc, -1, L.KER[Lv]) * (rad / 2.2) ** 1.2 * 1.6, (W, H))
    img *= L.EXP
    gw_ = w[inf] * (0.25 + 0.75 * val[inf]) ** 1.5 * (1 + 1.5 * tw[inf])
    gacc = np.stack([L.splat(u[inf] / 2, v[inf] / 2, gw_ * col[inf][:, c], H // 2, W // 2) for c in range(3)], -1)
    halo = cv2.GaussianBlur(gacc, (0, 0), 2.0) + cv2.GaussianBlur(gacc, (0, 0), 6.0) * 0.8
    img = img + cv2.resize(halo, (W, H)) * L.EXP * L.GLOW * (0.9 + 0.2 * p)
    img = img * L.vig
    Lm = luma(img)[..., None]
    Lt = Lm / (1 + Lm * 0.85)
    return np.clip(img * (Lt / np.maximum(Lm, 1e-5)) * 1.25, 0, 1)


GAIN_TD = 5.2
APER_TD = 10.0

if __name__ == '__main__' and sys.argv[1] == 'still':
    pd = float(sys.argv[2]); out = sys.argv[3]
    i = int(float(sys.argv[4]) * M.FPS) if len(sys.argv) > 4 else 125
    img = render(pd, i, drift=(M.trav[i] * 0.6, M.trav[i] * 0.3))
    cv2.imwrite(out, (img * 255 + 0.5).astype(np.uint8))
    print(out, 'mean', round(float(luma(img).mean()), 3))
