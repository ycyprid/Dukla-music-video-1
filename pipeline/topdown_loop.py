"""Top-down (72 deg) version of the benchmark: 15 s seamless loop, 25 fps, slow diagonal glide,
motion blur, gentle beat surges, smooth focus, per-dot glow, twinkle, fine dust, measured brightness levelling."""
import numpy as np
import topdown as T
L, M = T.L, T.M
from common import W, H, luma, Writer
cv2 = L.cv2

PITCH = np.radians(72.0)
CP, SP = np.cos(PITCH), np.sin(PITCH)
FPS, N, LOOP = M.FPS, M.N, L.LOOP
DIR = np.array([0.89, 0.45])                       # diagonal glide across the surface
GLIDE = 0.45                                       # fraction of the benchmark's travel: slow glide
tr = M.trav * GLIDE
DXT = round(tr[N] * DIR[0] / T.DX_) * T.DX_        # total drift per loop, whole dots
DZT = round(tr[N] * DIR[1] / T.DZ_) * T.DZ_
SX, SZ = int(round(DXT / T.DX_)), int(round(DZT / T.DZ_))
OX = tr / tr[N] * DXT; OZ = tr / tr[N] * DZT
IDX = np.arange(N + 1, dtype=np.float64)
PXg, PZg = T.build_grid(PITCH)
TX = 4.0
ZREF = T.CAMH - 1.15
ZF = ZREF + 0.25


def raw(xw, zw, tau):
    z = tau * 0.42
    u = xw * L.S; v = zw * L.S
    wx = L.sample(L.lattice(z * 0.6 + 10, 0.6), u, v)
    wy = L.sample(L.lattice(z * 0.6 + 40, 0.6), u, v)
    n = L.sample(L.lattice(z), u + 0.9 * wx + tau * 0.06, v + 0.9 * wy)
    fold = L.sample(L.lattice(z * 0.30 + 70, 0.35), u, v)
    return n, fold


def state(x):
    """surface at fractional frame x: camera-space dot positions + values (dots anchored to the ground)"""
    t = x / FPS
    tau = np.interp(x, IDX, M.taus)
    ox, oz = np.interp(x, IDX, OX), np.interp(x, IDX, OZ)
    ix, iz = np.floor(ox / T.DX_), np.floor(oz / T.DZ_)
    xw, zw = PXg + ix * T.DX_, PZg + iz * T.DZ_
    n, fold = raw(xw, zw, tau)
    k = np.clip((t - (LOOP - TX)) / TX, 0, 1); k = k * k * (3 - 2 * k)
    if k > 0:                                       # blend into the loop start (seamless)
        n2, f2 = raw(xw - DXT, zw - DZT, tau - M.TAU_T); th = k * np.pi / 2
        n = n * np.cos(th) + n2 * np.sin(th); fold = fold * np.cos(th) + f2 * np.sin(th)
    rid = (np.round(zw / T.DZ_).astype(np.int64), np.round(xw / T.DX_).astype(np.int64))
    return n, fold, PXg - (ox - ix * T.DX_), PZg - (oz - iz * T.DZ_), rid, k


def project(n, fold, xc, zc, p):
    Y = fold * 0.45 + L.shape(n, p * 0.3) * 0.95
    yc = Y - T.CAMH
    y2 = yc * CP + zc * SP; z2 = -yc * SP + zc * CP
    z2c = np.maximum(z2, 0.2)
    return W / 2 + L.FPX * xc / z2c, H / 2 - L.FPX * y2 / z2c, z2


def twinkle(rid, t, k):
    def tw(r, c):
        ph = L.hash01(r, c, 911); kk = 1 + np.floor(L.hash01(r, c, 4242) * 3)
        return np.maximum(np.sin(2 * np.pi * (kk * t / LOOP + ph)), 0) ** 24
    a = tw(*rid)
    if k > 0:                                       # glints hand over to the dot they'll be at loop start
        a = a * (1 - k) + tw(rid[0] - SZ, rid[1] - SX) * k
    return a


rng = np.random.default_rng(5)
ND = 90
DR = 10.0
DUX = rng.uniform(-DR / 2, DR / 2, ND); DUZ = rng.uniform(-DR / 2 + 1.5, DR / 2 + 1.5, ND)
DUY = rng.uniform(1.4, 3.6, ND)


def dust(t):
    s = DR * t / LOOP                               # one full wrap per loop -> seamless; drifts with the glide
    x = np.mod(DUX - DIR[0] * s + DR / 2, DR) - DR / 2
    z = np.mod(DUZ - 1.5 - DIR[1] * s + DR / 2, DR) - DR / 2 + 1.5
    yc = DUY - T.CAMH
    y2 = yc * CP + z * SP; z2 = -yc * SP + z * CP
    z2c = np.maximum(z2, 0.2)
    return W / 2 + L.FPX * x / z2c, H / 2 - L.FPX * y2 / z2c, z2


def render(i, gain=1.0, blur=True):
    t = i / FPS; p = M.PULSES[i]
    n, fold, xc, zc, rid, k = state(i)
    val = L.shape(n, p)
    u, v, z2 = project(n, fold, xc, zc, p)
    ok = z2 > 0.2; z2c = np.maximum(z2, 0.2)
    col = L.PAL[((0.10 + 0.90 * val) * 255).astype(np.uint8)]
    fog = np.clip((z2 - ZREF) / (T.CAMH + 1.5 - ZREF), 0, 1)[:, None] ** 0.9
    col = col * (1 - fog * 0.35) + L.DEEP * fog * 0.35
    w = (0.38 + 0.62 * val) * ok * np.minimum(1.0, (ZREF / z2c) ** 1.2) * np.exp(-np.maximum(z2 - ZREF, 0) / 9.0)
    w = w * 2.0 * T.GAIN_TD * gain
    tw = twinkle(rid, t, k) * (0.4 + 0.6 * val)
    w = w * (1 + 1.5 * tw)
    r = np.maximum(L.FPX * 0.004 / z2c, T.APER_TD * np.abs(1 / z2c - 1 / ZF))
    inf = ok & (u > -60) & (u < W + 60) & (v > -60) & (v < H + 60)
    U, V, C, Wt, Rr = u[inf], v[inf], col[inf], w[inf], r[inf]
    if blur:                                        # 180-degree shutter
        n1, f1, xc1, zc1, _, _ = state(i + 0.5)
        if xc1.shape == xc.shape:
            u1, v1, _ = project(n1, f1, xc1, zc1, p)
            du, dv = (u1 - u)[inf], (v1 - v)[inf]
            jump = np.hypot(du, dv) > 30            # grid re-snap between samples: skip those
            du[jump] = 0; dv[jump] = 0
            fast = np.hypot(du, dv) > 0.75
            K = 7; sl = (np.arange(K) + 0.5) / K
            U = np.concatenate([U[~fast], (U[fast][None] + du[fast][None] * sl[:, None]).ravel()])
            V = np.concatenate([V[~fast], (V[fast][None] + dv[fast][None] * sl[:, None]).ravel()])
            C = np.concatenate([C[~fast], np.tile(C[fast], (K, 1))])
            Wt = np.concatenate([Wt[~fast], np.tile(Wt[fast] / K, K)])
            Rr = np.concatenate([Rr[~fast], np.tile(Rr[fast], K)])
    du_, dv_, dz2 = dust(t)
    dr = np.maximum(30.0 * np.abs(1 / np.maximum(dz2, 0.2) - 1 / ZF) * 0.55, 1.0)
    dw = 0.55 * (dz2 > 0.2)
    U = np.concatenate([U, du_]); V = np.concatenate([V, dv_])
    C = np.concatenate([C, L.PAL[np.full(ND, int(0.80 * 255))]]); Wt = np.concatenate([Wt, dw]); Rr = np.concatenate([Rr, dr])
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
    return np.clip(img * ((Lm / (1 + Lm * 0.85)) / np.maximum(Lm, 1e-5)) * 1.25, 0, 1)


if __name__ == '__main__':
    print('drift per loop', DXT, DZT, flush=True)
    samp = np.arange(0, N, 3)
    meas = np.array([cv2.resize(render(i, 1.0, blur=False), (160, 90), interpolation=cv2.INTER_AREA).mean() for i in samp])
    b = np.interp(np.arange(N), np.append(samp, N), np.append(meas, meas[0]))
    kk = np.exp(-0.5 * (np.arange(-24, 25) / 6.0) ** 2); kk /= kk.sum()
    bs = np.real(np.fft.ifft(np.fft.fft(b) * np.fft.fft(np.roll(np.pad(kk, (0, N - kk.size)), -24))))
    gain = np.clip((np.median(bs) / bs) ** 1.15, 0.55, 1.8)
    print('measured brightness range x%.2f, gain %.2f-%.2f' % (bs.max() / bs.min(), gain.min(), gain.max()), flush=True)
    wr = Writer('out/D_topdown_loop15_25fps.mp4', 'out/stills_td', [0, 60, 125, 190, 250, 320, 374])
    for i in range(N):
        wr.write(render(i, gain[i]))
    wr.close()
