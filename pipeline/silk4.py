"""Texture after the reference close-up: smooth dark/light masses; the light masses have scalloped lumpy edges,
soft relief on the lumps, and fine fur combed ACROSS the contours (pixels propagated along the image gradient),
bleeding from the lights into the darks. Darks stay smooth. D palette, receding plane, specks, 146 BPM loop."""
import sys
import numpy as np, cv2, opensimplex
from common import W, H, luma, gradient_lut, Writer
import common
import silk as S
import silk3 as L3

N, LOOP, FPS = S.N, S.LOOP, S.FPS
TW, TH = S.TW, S.TH
PAL = L3.PAL
rng = np.random.default_rng(8)
FUR = cv2.GaussianBlur(rng.random((TH, TW)).astype(np.float32), (0, 0), 0.9)
FUR = (FUR - FUR.mean()) / FUR.std()
yy, xx = np.mgrid[0:TH, 0:TW].astype(np.float32)


def up(a):
    return cv2.resize(a.astype(np.float32), (TW, TH), interpolation=cv2.INTER_CUBIC)


def streaks(dx, dy, steps=16, ds=1.4):
    """fine fibres along the gradient direction (across contours)"""
    acc = np.zeros((TH, TW), np.float32); ws = 0.0
    for d in (1, -1):
        px, py = xx.copy(), yy.copy()
        for s in range(steps):
            w = np.exp(-(s / steps) ** 2 * 2.0)
            acc += w * cv2.remap(FUR, px, py, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT); ws += w
            ux = cv2.remap(dx, px, py, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            uy = cv2.remap(dy, px, py, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            px += d * ux * ds; py += d * uy * ds
    return acc / ws


def render(i):
    ph = i / N; t = ph * LOOP
    pulse = S.beat_pulse(t)
    val, hgt, rxw, qyw = L3.surface(ph)
    # swirl composition: one spiral arm curling round a light centre, warped by the liquid, winding 1 turn/loop
    cx, cy = 1.5 + 0.12 * np.cos(2 * np.pi * ph), 1.42 + 0.08 * np.sin(2 * np.pi * ph)
    X = L3.GX + 0.28 * rxw - cx; Y = (L3.GY + 0.28 * qyw - cy) * 1.25
    r = np.sqrt(X * X + Y * Y) + 0.05; th = np.arctan2(Y, X)
    sp = np.sin(th - 3.6 * np.log(r + 0.15) - 2 * np.pi * ph) + 0.18
    core = np.exp(-(r / 0.22) ** 2) * 0.7                                     # bright heart of the swirl
    fade = np.exp(-(r / 2.2) ** 2)
    val = (sp * 0.85 * (0.35 + 0.65 * fade) + core * 1.1) * 1.0 + val * 0.35
    V = (val - np.median(val)) / (np.percentile(val, 95) - np.percentile(val, 5) + 1e-6)
    V = up(V)
    # lumps: mid-frequency periodic noise, travelling with the warp -> scalloped edges of the light masses
    lump = L3.samp(L3.lat(4.2, 61.0, ph), *np.meshgrid(L3.gx * 0.55, L3.gy * 0.55)) * 0.65 + \
           L3.samp(L3.lat(8.5, 67.0, ph), *np.meshgrid(L3.gx * 0.55, L3.gy * 0.55)) * 0.35
    lump = cv2.GaussianBlur(up(lump), (0, 0), 5.0)                        # smooth after upsampling: no grid in the relief
    light = np.clip((V + 0.15) / 0.5, 0, 1)
    Vb = V + 0.16 * lump * light                                      # bumps only where it's light
    v0 = np.clip(0.5 + 1.15 * Vb, 0, 1); v0 = v0 * v0 * (3 - 2 * v0)
    v0 = cv2.GaussianBlur(v0, (0, 0), 2.0)
    # gradient direction (across the contours)
    gy_, gx_ = np.gradient(cv2.GaussianBlur(v0, (0, 0), 6.0))
    gn = np.sqrt(gx_ * gx_ + gy_ * gy_) + 1e-6
    dx, dy = (gx_ / gn).astype(np.float32), (gy_ / gn).astype(np.float32)
    # varied fibre length: slow noise mask mixes short and long fibres across the surface
    lm = up(L3.samp(L3.lat(1.7, 73.0, ph), L3.GX * 0.5, L3.GY * 0.5))
    lm = np.clip(0.5 + 0.9 * lm, 0, 1); lm = lm * lm * (3 - 2 * lm)
    fs_ = streaks(dx, dy, steps=7, ds=1.2); fs_ /= fs_.std() + 1e-6
    fl_ = streaks(dx, dy, steps=28, ds=1.7); fl_ /= fl_.std() + 1e-6
    fib = fs_ * (1 - lm) + fl_ * lm
    slope = np.clip(gn / (np.percentile(gn, 70) + 1e-6), 0, 1)        # no clear direction (flat) -> no fibres
    fib = fib * slope
    # propagation: light pixels pulled outward across the edge by the fibre noise -> hairy, combed edges
    L = (6.0 + 22.0 * lm) * np.clip(fib * 0.6 + 0.5, 0, 1.6)        # short fringe or long combed strands
    v = cv2.remap(v0.astype(np.float32), (xx + dx * L).astype(np.float32), (yy + dy * L).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    edge = np.clip(1 - np.abs(v - 0.5) / 0.5, 0, 1)
    v = np.clip(v + fib * (0.035 * v + 0.06 * edge), 0, 1)
    dark_smooth = cv2.GaussianBlur(v, (0, 0), 3.0)
    v = v * np.clip(v * 1.6, 0, 1) + dark_smooth * (1 - np.clip(v * 1.6, 0, 1))   # darks stay smooth
    # soft relief on the lumps (lights only): key light from the upper left
    ry, rx = np.gradient(cv2.GaussianBlur(lump * light + v0 * 0.6, (0, 0), 2.5))
    shade = np.clip(1 + (-rx * 0.6 - ry * 0.8) * 55, 0.55, 1.45)
    shade = 1 + (shade - 1) * light
    lum = np.clip(v * shade, 0, 1)
    col = PAL[(lum * 255).astype(np.uint8)]
    sheen = np.clip(shade - 1.15, 0, None) * light * 0.9               # soft wet highlight on lump tops
    col = col + PAL[255][None, None] * (sheen * (0.8 + 0.4 * pulse))[..., None]
    col = col * (0.96 + 0.08 * pulse)
    scr = cv2.warpPerspective(col, S.HM, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    far = cv2.GaussianBlur(scr, (0, 0), 3.5); near = cv2.GaussianBlur(scr, (0, 0), 1.5)
    d = S.DEPTH[..., None]
    swt = np.exp(-((d - 0.62) / 0.28) ** 2)
    img = scr * swt + (far * (d < 0.62) + near * (d >= 0.62)) * (1 - swt)
    img = img * (0.55 + 0.45 * np.clip(d * 1.5, 0, 1))
    img = S.specks(img, ph, pulse)
    hi = np.clip(img - 0.78, 0, None)
    img = img + cv2.resize(cv2.GaussianBlur(cv2.resize(hi, (W // 4, H // 4)), (0, 0), 3), (W, H)) * (0.3 + 0.15 * pulse)
    Lm = luma(img)[..., None]
    return np.clip(img * ((Lm / (1 + Lm * 0.45)) / np.maximum(Lm, 1e-5)) * 1.15, 0, 1)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'still':
        cv2.imwrite(sys.argv[3], (render(int(sys.argv[2])) * 255 + 0.5).astype(np.uint8))
    else:
        common.FPS = FPS
        wr = Writer('out/E_swirl_loop.mp4', 'out/stills_swirl', [0, 80, 160, 240, 328])
        for i in range(N):
            wr.write(render(i))
        wr.close()
