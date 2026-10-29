"""Liquid v6: fur-loop field and balance, calmer motion, edges brightest (yellow ribbon, white core), glossy liquid
surface with white sparkles on the ridges, thermal glow. D palette, receding plane, specks, seamless 32 beats @146."""
import sys
import numpy as np, cv2
from common import W, H, luma, gradient_lut, Writer
import common
import silk as S
import silk3 as L3

L3.RAD = 0.62                                   # half the change per loop -> calm, rolling motion
N, LOOP, FPS = S.N, S.LOOP, S.FPS
TW, TH = S.TW, S.TH
hx = S.hexrgb
BASE = gradient_lut([(0.00, hx('0A0004')), (0.24, hx('2A0412')), (0.42, hx('5C0A22')), (0.54, hx('962026')),
                     (0.64, hx('D4542A')), (0.80, hx('E8882E')), (1.00, hx('F2A63E'))])
YEL = np.array(hx('FFC84A')[::-1], np.float32) / 255
WHT = np.array(hx('FFF7E4')[::-1], np.float32) / 255
rng = np.random.default_rng(6)
NSP = 9000
SPK_X = rng.integers(0, TW, NSP); SPK_Y = rng.integers(0, TH, NSP)
SPK_P = rng.random(NSP).astype(np.float32); SPK_K = rng.integers(3, 9, NSP).astype(np.float32)
GRAIN = [cv2.GaussianBlur(rng.standard_normal((H, W)).astype(np.float32), (0, 0), 0.6) for _ in range(8)]


def up(a):
    return cv2.resize(a.astype(np.float32), (TW, TH), interpolation=cv2.INTER_CUBIC)


def render(i):
    ph = i / N; t = ph * LOOP
    pulse = S.beat_pulse(t)
    val, hgt, rxw, qyw = L3.surface(ph)
    V = up((val - np.median(val)) / (np.percentile(val, 95) - np.percentile(val, 5) + 1e-6))
    V = cv2.GaussianBlur(V, (0, 0), 3.0)
    v = np.clip(0.5 + 1.15 * V, 0, 1); v = v * v * (3 - 2 * v)                     # fur-loop balance
    # glossy liquid surface: height = value, smooth normals, satin key + sharp specular
    hs = cv2.GaussianBlur(v, (0, 0), 6.0)
    gy, gx = np.gradient(hs)
    k = 260.0
    nx, ny = -gx * k, -gy * k; nz = np.ones_like(hs)
    nl = np.sqrt(nx * nx + ny * ny + 1); nx /= nl; ny /= nl; nz /= nl
    Lk = np.array([-0.45, -0.6, 0.66]); Lk /= np.linalg.norm(Lk)
    diff = np.clip(nx * Lk[0] + ny * Lk[1] + nz * Lk[2], 0, 1)
    Hh = Lk + np.array([0, 0, 1.0]); Hh /= np.linalg.norm(Hh)
    nh = np.clip(nx * Hh[0] + ny * Hh[1] + nz * Hh[2], 0, 1)
    # sparkle: tiny moving ripples on the surface break the specular into glints
    gmask = nh ** 40 * np.clip(v * 1.5, 0, 1)                                     # where the liquid catches the light
    tw = np.maximum(np.sin(2 * np.pi * (SPK_K * ph + SPK_P)), 0) ** 6               # each point twinkles, periodic
    spark = np.zeros((TH, TW), np.float32)
    spark[SPK_Y, SPK_X] = tw * gmask[SPK_Y, SPK_X]
    spark = cv2.GaussianBlur(spark, (0, 0), 0.9) * 9 + cv2.GaussianBlur(spark, (0, 0), 3.0) * 6
    spec = spark + nh ** 200 * 0.25 * np.clip(v * 1.5, 0, 1)
    base = BASE[(np.clip(v * (0.55 + 0.6 * diff), 0, 1) * 255).astype(np.uint8)]
    # edges brightest: yellow ribbon with a white-hot core along the light/dark boundary
    slope = np.sqrt(gx * gx + gy * gy)
    em = np.clip(slope / (np.percentile(slope, 92) + 1e-6), 0, 1)
    d = v - 0.6
    ribbon = np.exp(-(d / 0.13) ** 2) * em
    core = np.exp(-(d / 0.035) ** 2) * em
    col = base + YEL[None, None] * (ribbon * 0.9)[..., None] + WHT[None, None] * (core * 0.75)[..., None]
    col = col + WHT[None, None] * (spec * (1.3 + 0.5 * pulse))[..., None]
    col = col * (0.95 + 0.08 * pulse)
    scr = cv2.warpPerspective(col, S.HM, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    far = cv2.GaussianBlur(scr, (0, 0), 3.0); near = cv2.GaussianBlur(scr, (0, 0), 1.3)
    dd = S.DEPTH[..., None]
    swt = np.exp(-((dd - 0.62) / 0.3) ** 2)
    img = scr * swt + (far * (dd < 0.62) + near * (dd >= 0.62)) * (1 - swt)
    img = img * (0.6 + 0.4 * np.clip(dd * 1.5, 0, 1))
    # thermal glow around the bright parts
    hi = np.clip(img - 0.55, 0, None)
    sm = cv2.resize(hi, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
    img = img + cv2.resize(cv2.GaussianBlur(sm, (0, 0), 3) * 0.6 + cv2.GaussianBlur(sm, (0, 0), 10) * 0.5, (W, H)) * (0.8 + 0.3 * pulse)
    img = S.specks(img, ph, pulse)
    Lm = luma(np.clip(img, 0, 1))[..., None]
    img = np.clip(img + GRAIN[i % 8][..., None] * 0.022 * (0.3 + Lm), 0, None)
    Lm = luma(img)[..., None]
    return np.clip(img * ((Lm / (1 + Lm * 0.5)) / np.maximum(Lm, 1e-5)) * 1.15, 0, 1)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'still':
        cv2.imwrite(sys.argv[3], (render(int(sys.argv[2])) * 255 + 0.5).astype(np.uint8))
    else:
        common.FPS = FPS
        wr = Writer('out/E_liquid_v6.mp4', 'out/stills_v6', [0, 80, 160, 240, 328])
        for i in range(N):
            wr.write(render(i))
        wr.close()
