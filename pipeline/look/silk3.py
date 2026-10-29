"""Liquid v3: no streaks, no vortices. Smooth domain-warped noise + travelling swells = soft waves; the surface
reflects a warm sky (burgundy -> gold -> cream) with Fresnel and soft glints = water-like. D palette for the
base values, same specks / receding plane / depth of field / 146 BPM pulse. Seamless 32-beat loop."""
import sys
import numpy as np, cv2, opensimplex
from common import W, H, luma, gradient_lut, Writer
import common
import silk as S

opensimplex.seed(52)
N, LOOP, FPS = S.N, S.LOOP, S.FPS
TW, TH = S.TW, S.TH
RAD = 1.2
GW_, GH_ = 300, 205                                    # evaluation grid (smooth field, upsampled)
gx = np.linspace(0, 3.0, GW_).astype(np.float32); gy = np.linspace(0, 2.05, GH_).astype(np.float32)
GX, GY = np.meshgrid(gx, gy)
MX = 1.6
lx = np.linspace(-MX, 3.0 + MX, 170); ly = np.linspace(-MX, 2.05 + MX, 130)

PAL = gradient_lut([(0.00, S.hexrgb('16000A')), (0.14, S.hexrgb('5A0620')), (0.28, S.hexrgb('8A0A34')),
                    (0.40, S.hexrgb('C2183F')), (0.51, S.hexrgb('D8542C')), (0.62, S.hexrgb('E8A322')),
                    (0.74, S.hexrgb('F6B93A')), (0.86, S.hexrgb('FFDC96')), (1.00, S.hexrgb('FFF6E6'))])
SKY = gradient_lut([(0.00, S.hexrgb('2A0010')), (0.30, S.hexrgb('8A0A34')), (0.52, S.hexrgb('D8542C')),
                    (0.70, S.hexrgb('F2A93A')), (0.86, S.hexrgb('FFE6B0')), (1.00, S.hexrgb('FFF8EC'))])


def lat(sc, off, ph):
    a = 2 * np.pi * ph
    return opensimplex.noise4array(lx * sc + off, ly * sc, np.array([RAD * np.cos(a) + off]),
                                   np.array([RAD * np.sin(a)]))[0, 0].astype(np.float32)


def samp(L, x, y):
    return cv2.remap(L, ((x - lx[0]) / (lx[1] - lx[0])).astype(np.float32),
                     ((y - ly[0]) / (ly[1] - ly[0])).astype(np.float32), cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)


def surface(ph):
    """layered domain warp (smooth folds, no vortices) + travelling swells"""
    a1, a2 = lat(0.8, 0.0, ph), lat(0.8, 7.0, ph)
    qx, qy = samp(a1, GX, GY), samp(a2, GX, GY)
    b1, b2 = lat(0.9, 13.0, ph), lat(0.9, 19.0, ph)
    rx = samp(b1, GX + 1.1 * qx, GY + 1.1 * qy); ry = samp(b2, GX + 1.1 * qx, GY + 1.1 * qy)
    c = lat(0.75, 29.0, ph)
    val = samp(c, GX + 1.3 * rx, GY + 1.3 * ry)                    # the liquid value field (light / dark masses)
    # swells: long waves travelling across, bent by the warp; integer cycles per loop -> seamless
    ang = 0.35
    d = GX * np.cos(ang) + GY * np.sin(ang)
    sw = np.sin(2 * np.pi * (d * 1.1 - 3 * ph) + 2.2 * qx) * 0.55 + np.sin(2 * np.pi * (d * 2.3 + GY * 0.6 - 5 * ph) + 1.6 * ry) * 0.25
    hgt = val * 0.9 + sw * 0.35
    return val, hgt, rx, qy


def render(i):
    ph = i / N; t = ph * LOOP
    pulse = S.beat_pulse(t)
    val, hgt, rxw, qyw = surface(ph)
    up = lambda a: cv2.resize(a.astype(np.float32), (TW, TH), interpolation=cv2.INTER_CUBIC)
    # everything below at full texture resolution (crisp ripples / sparkles)
    val, hgt, rxw, qyw = up(val), up(hgt), up(rxw), up(qyw)
    # fine ripples: periodic noise (not sine products -> no mesh pattern), bent along the warp
    rpl = lat(3.2, 41.0, ph) * 0.6 + lat(5.5, 47.0, (ph * 2) % 1.0) * 0.4
    FX = (np.arange(TW, dtype=np.float32) / TW * 3.0)[None, :].repeat(TH, 0)
    FY = (np.arange(TH, dtype=np.float32) / TH * 2.05)[:, None].repeat(TW, 1)
    rp = samp(rpl, FX * 0.55 + 0.25 * rxw, FY * 0.55 + 0.25 * qyw)
    # values: broad, balanced light/dark masses
    b = (val - np.median(val)) / (np.percentile(val, 95) - np.percentile(val, 5) + 1e-6)
    v = np.clip(0.5 + 1.1 * b, 0, 1); v = v * v * (3 - 2 * v)
    # smooth normals (large scale only -> rolling water, not grooves)
    sc = TW / GW_
    dyL, dxL = np.gradient(cv2.GaussianBlur(hgt, (0, 0), 6.0))         # big swells: broad, rolling shading
    dyS, dxS = np.gradient(cv2.GaussianBlur(rp, (0, 0), 1.2))          # small ripples: sparkle
    nx = -(dxL * 9.0 * sc + dxS * 2.0 * sc); ny = -(dyL * 9.0 * sc + dyS * 2.0 * sc); nz = np.ones_like(hgt)
    nl = np.sqrt(nx * nx + ny * ny + nz * nz); nx /= nl; ny /= nl; nz /= nl
    # view from a low camera looking across the plane
    vx_, vy_, vz_ = 0.0, 0.55, 0.835
    ndv = np.clip(nx * vx_ + ny * vy_ + nz * vz_, 0, 1)
    rx = 2 * ndv * nx - vx_; ry = 2 * ndv * ny - vy_; rz = 2 * ndv * nz - vz_
    # environment: warm sky by reflection elevation, a soft bright band (the "sun" streak) + glints
    elev = np.clip(0.5 + 0.5 * (rz * 0.9 - ry * 0.6), 0, 1)
    env = SKY[(elev * 255).astype(np.uint8)]
    sun = np.array([0.25, -0.55, 0.80]); sun /= np.linalg.norm(sun)
    sd = np.clip(rx * sun[0] + ry * sun[1] + rz * sun[2], 0, 1)
    glint = sd ** 900 * 1.6 + sd ** 60 * 0.10                       # small sharp sparkles + faint sheen
    fres = 0.06 + 0.38 * (1 - ndv) ** 3
    base = PAL[(v * 255).astype(np.uint8)]
    fr3 = fres[..., None]
    col = base * (1 - fr3 * 0.5) + env * fr3 * 0.95
    col = col + PAL[255][None, None] * (glint * (0.8 + 0.6 * pulse))[..., None]
    col = col * (0.96 + 0.08 * pulse)
    scr = cv2.warpPerspective(col, S.HM, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    far = cv2.GaussianBlur(scr, (0, 0), 3.5); near = cv2.GaussianBlur(scr, (0, 0), 1.6)
    d = S.DEPTH[..., None]
    swt = np.exp(-((d - 0.62) / 0.26) ** 2)
    img = scr * swt + (far * (d < 0.62) + near * (d >= 0.62)) * (1 - swt)
    img = img * (0.5 + 0.5 * np.clip(d * 1.5, 0, 1))
    img = S.specks(img, ph, pulse)
    hi = np.clip(img - 0.75, 0, None)
    img = img + cv2.resize(cv2.GaussianBlur(cv2.resize(hi, (W // 4, H // 4)), (0, 0), 3), (W, H)) * (0.30 + 0.15 * pulse)
    Lm = luma(img)[..., None]
    return np.clip(img * ((Lm / (1 + Lm * 0.45)) / np.maximum(Lm, 1e-5)) * 1.18, 0, 1)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'still':
        cv2.imwrite(sys.argv[3], (render(int(sys.argv[2])) * 255 + 0.5).astype(np.uint8))
    else:
        common.FPS = FPS
        wr = Writer('out/E_liquid_loop.mp4', 'out/stills_liquid', [0, 80, 160, 240, 328])
        for i in range(N):
            wr.write(render(i))
        wr.close()
