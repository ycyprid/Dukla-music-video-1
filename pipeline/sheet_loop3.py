"""3D dot sheet, 15 s seamless loop, v3 (25 fps).
vs v2: motion blur on the dots (180-degree shutter, multi-sample for fast dots), gentler beat surges on
the terrain shape and the glide (light/glow/twinkle keep their pulse), 25 fps, brightness levelling measured
from a quick real render pass. Everything else (look, palette, twinkle, dust, seams/edge fixes) from v2."""
import sys, os
import numpy as np
import common
common.FPS = 25
import sheet_loop2 as L                     # shared look: palette, lattice/field, twinkle, dust, splat, focus levels
from common import W, H, luma, soft_pulse, Writer, BEAT

FPS = 25
LOOP = L.LOOP
N = int(LOOP * FPS)                         # 375 frames; 1 beat = 15 frames exactly
PULSES = np.array([soft_pulse(i / FPS) for i in range(N)])
PM = PULSES.mean()
SURGE_T, SURGE_G = 0.5, 0.08                # was 1.6 (shape) and 0.30 (glide): much gentler
taus, trav = [0.0], [0.0]
for i in range(N):
    p = PULSES[i]
    taus.append(taus[-1] + (1.0 + (1.6 - SURGE_T) * PM + SURGE_T * p) / FPS)     # same average speed as v2
    trav.append(trav[-1] + L.SPEED * (1.0 + (0.30 - SURGE_G) * PM + SURGE_G * p) / FPS)
taus = np.array(taus); trav = np.array(trav)
TAU_T = taus[N]
D = round(trav[N] / L.DZW) * L.DZW
ROWS_PER_LOOP = int(round(D / L.DZW))
trav = trav * D / trav[N]
IDX = np.arange(N + 1, dtype=np.float64)
TX = 4.0


def field_at(x, wz):
    """field at fractional frame x, sampled on world rows PZ + wz"""
    t = x / FPS
    tau = np.interp(x, IDX, taus)
    n, fold = L.field_raw(tau, wz)
    k = np.clip((t - (LOOP - TX)) / TX, 0, 1); k = k * k * (3 - 2 * k)
    if k > 0:
        n2, f2 = L.field_raw(tau - TAU_T, wz - D); th = k * np.pi / 2
        n = n * np.cos(th) + n2 * np.sin(th); fold = fold * np.cos(th) + f2 * np.sin(th)
    return n, fold


def project(valg, fold, frac, t):
    Y = fold * 0.55 + valg * 0.30
    camx = 0.30 * np.sin(2 * np.pi * t / LOOP)
    yc = Y - L.CAMY; xc = L.PX - camx; zc = L.PZ - frac
    y2 = yc * L.CP + zc * L.SP; z2 = -yc * L.SP + zc * L.CP
    z2c = np.maximum(z2, 0.15)
    return L.W / 2 + L.FPX * xc / z2c, L.H / 2 - L.FPX * y2 / z2c, z2, camx


def render(i, gain, blur=True):
    t = i / FPS
    p = PULSES[i]
    tr = trav[i]
    wz = np.floor(tr / L.DZW + 1e-6) * L.DZW
    row_id = np.round((L.PZ + wz) / L.DZW).astype(np.int64) % ROWS_PER_LOOP
    n, fold = field_at(i, wz)
    val = L.shape(n, p)                     # colour / brightness: full beat swell
    valg = L.shape(n, p * 0.3)              # height: gentle swell, so hilltops don't lurch
    u, v, z2, camx = project(valg, fold, tr - wz, t)
    ok = z2 > 0.15
    z2c = np.maximum(z2, 0.15)
    col = L.PAL[((0.10 + 0.90 * val) * 255).astype(np.uint8)]
    fog = np.clip((z2 - 2.5) / 12.0, 0, 1)[:, None] ** 0.8
    col = col * (1 - fog * 0.7) + L.DEEP * fog * 0.7
    w = (0.30 + 0.70 * val) * ok * 2.0 * np.minimum(1.0, (1.3 / z2c) ** 2) \
        * np.exp(-np.maximum(z2 - 1.3, 0) / 6.5) * L.NEAR_GAIN
    w = w * (1 - np.clip((z2 - 15.0) / 5.0, 0, 1))
    tw = L.twinkle(row_id, t) * (0.4 + 0.6 * val)
    w = w * gain * (1 + 1.5 * tw)
    r = np.maximum(L.FPX * 0.004 / z2c, L.APER * np.abs(1 / z2c - 1 / L.ZF))
    inf = (u > -60) & (u < W + 60) & (v > -60) & (v < H + 60) & ok

    # motion blur: where each dot travels during the open half of the shutter (frame i -> i+0.5)
    U, V, C, Wt, Rr = u[inf], v[inf], col[inf], w[inf], r[inf]
    if blur:
        x1 = i + 0.5
        n1, fold1 = field_at(x1, wz)
        u1, v1, _, _ = project(L.shape(n1, PULSES[i] * 0.3), fold1, np.interp(x1, IDX, trav) - wz, x1 / FPS)
        du, dv = (u1 - u)[inf], (v1 - v)[inf]
        disp = np.hypot(du, dv)
        fast = disp > 0.75
        K = 7
        sl = (np.arange(K) + 0.5) / K
        Uf = (U[fast][None] + du[fast][None] * sl[:, None]).ravel()
        Vf = (V[fast][None] + dv[fast][None] * sl[:, None]).ravel()
        U = np.concatenate([U[~fast], Uf]); V = np.concatenate([V[~fast], Vf])
        C = np.concatenate([C[~fast], np.tile(C[fast], (K, 1))])
        Wt = np.concatenate([Wt[~fast], np.tile(Wt[fast] / K, K)])
        Rr = np.concatenate([Rr[~fast], np.tile(Rr[fast], K)])

    # dust (fine specks, streaming toward camera)
    DZd, dfade = L.dust_z(t)
    ph = 2 * np.pi * L.DK * t / LOOP
    dx = L.DX + 0.05 * np.sin(ph + L.DPH) - camx
    dy = L.DY + 0.06 * np.sin(ph + L.DPH * 1.3) - L.CAMY
    dz2 = -dy * L.SP + DZd * L.CP; dy2 = dy * L.CP + DZd * L.SP
    dz2c = np.maximum(dz2, 0.15)
    du_ = W / 2 + L.FPX * dx / dz2c; dv_ = H / 2 - L.FPX * dy2 / dz2c
    dr = np.maximum(30.0 * np.abs(1 / dz2c - 1 / L.ZF) * 0.55, 1.0)
    dcol = L.PAL[np.full(L.ND, int(0.80 * 255))]
    dw = 0.55 * (dz2 > 0.15) * dfade * (0.6 + 0.4 * np.sin(L.DPH + 2 * np.pi * t / LOOP) ** 2)
    U = np.concatenate([U, du_]); V = np.concatenate([V, dv_])
    C = np.concatenate([C, dcol]); Wt = np.concatenate([Wt, dw]); Rr = np.concatenate([Rr, dr])

    # smooth focus levels (no seams)
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
            img += L.cv2.GaussianBlur(acc, (0, 0), rad) * (rad / 0.7) ** 1.2
        else:
            acc = np.stack([L.splat(U[m] / 2, V[m] / 2, ww * C[m, c], H // 2, W // 2) for c in range(3)], -1)
            img += L.cv2.resize(L.cv2.filter2D(acc, -1, L.KER[Lv]) * (rad / 2.2) ** 1.2 * 1.6, (W, H))
    img *= L.EXP
    hy = H / 2 - L.FPX * np.tan(L.PITCH - np.arctan(L.CAMY / 15.5))
    yy = np.arange(H, dtype=np.float32)[:, None, None]
    img += L.HAZE * 0.025 * np.exp(-((yy - hy) / (H * 0.04)) ** 2)
    # per-point glow
    gw_ = w[inf] * (0.25 + 0.75 * val[inf]) ** 1.5 * (1 + 1.5 * tw[inf])
    ui, vi, ci = u[inf], v[inf], col[inf]
    gacc = np.stack([L.splat(ui / 2, vi / 2, gw_ * ci[:, c], H // 2, W // 2) for c in range(3)], -1)
    halo = L.cv2.GaussianBlur(gacc, (0, 0), 2.0) + L.cv2.GaussianBlur(gacc, (0, 0), 6.0) * 0.8
    img = img + L.cv2.resize(halo, (W, H)) * L.EXP * L.GLOW * (0.9 + 0.2 * p)
    img = img * L.vig
    Lm = luma(img)[..., None]
    Lt = Lm / (1 + Lm * 0.85)
    return np.clip(img * (Lt / np.maximum(Lm, 1e-5)) * 1.25, 0, 1)


if __name__ == '__main__':
    # pass 1: real (quick, unblurred) render of every 3rd frame -> measured brightness, loop-wrapped smoothing
    samp = np.arange(0, N, 3)
    meas = np.array([L.cv2.resize(render(i, 1.0, blur=False), (160, 90), interpolation=L.cv2.INTER_AREA).mean() for i in samp])
    b = np.interp(np.arange(N), np.append(samp, N), np.append(meas, meas[0]))
    sig = 6.0
    k = np.exp(-0.5 * (np.arange(-24, 25) / sig) ** 2); k /= k.sum()
    bs = np.real(np.fft.ifft(np.fft.fft(b) * np.fft.fft(np.roll(np.pad(k, (0, N - k.size)), -24))))
    gain = np.clip((np.median(bs) / bs) ** 1.15, 0.55, 1.8)
    np.save('out/loop3_gain.npy', np.stack([b, bs, gain]))
    print('measured brightness range x%.2f, gain %.2f-%.2f' % (bs.max() / bs.min(), gain.min(), gain.max()), flush=True)
    # pass 2: full render with motion blur
    wr = Writer('out/D_sheet_loop15_25fps.mp4', 'out/stills_loop3', [0, 40, 100, 190, 280, 350, 374])
    for i in range(N):
        wr.write(render(i, gain[i]))
    wr.close()
