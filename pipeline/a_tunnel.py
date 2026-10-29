"""Direction A - Golden Tunnel: canopy footage wrapped onto an infinite tunnel, couple at the vanishing point."""
import sys
from common import *

DUR = float(sys.argv[1]) if len(sys.argv) > 1 else 10
MODE = sys.argv[2] if len(sys.argv) > 2 else 'clean'
canopy = Clip('src/h2.mp4')
couple = Clip('src/h4.mp4')

y, x = np.mgrid[0:H, 0:W].astype(np.float32)
X = (x - W / 2) / H
Y = (y - H / 2) / H
R = np.sqrt(X * X + Y * Y) + 1e-4
A = np.arctan2(Y, X)


def mirror(v):
    v = np.mod(v, 2.0)
    return 1 - np.abs(1 - v)


vig = vignette(k=0.6)
glow = np.array([0.55, 0.85, 1.0], np.float32)  # BGR warm gold
prev = None
depth = 0.0
N = int(DUR * FPS)
wr = Writer(f'out/A_tunnel_{MODE}.mp4', f'out/stills_A_{MODE}', [45, 150, 255])
core_r = 0.15
for i in range(N):
    t = i / FPS
    _, swell = beat(t)
    kick = soft_pulse(t)
    depth += (0.11 + 0.22 * kick + 0.05 * swell) / FPS   # integrated -> smooth surges on beats
    Rk = R * (1 - 0.015 * kick)
    u = mirror((A / np.pi + 1) * 2.0 + 0.03 * t)                 # 4 mirrored repeats around the ring
    v = mirror(0.28 / Rk + depth)
    tex = canopy.get(i).astype(np.float32) / 255
    img = cv2.remap(tex, (u * (W - 1)).astype(np.float32), (v * (H - 1)).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    img = warm_split(img, 0.55)
    img = np.clip((img - 0.5) * 1.25 + 0.42, 0, 1) ** 1.15
    # depth fog -> warm light at the far end
    fog = np.clip(1 - (Rk - core_r) / 0.22, 0, 1)[..., None] ** 2.5
    img = img * (1 - fog) + glow * fog * (0.62 + 0.15 * kick)
    # couple at the vanishing point
    c = couple.get(i).astype(np.float32) / 255
    cs = int(H * 0.44)
    cx, cy = int(W * 0.5), int(H * 0.52)
    crop = c[cy - cs // 2: cy + cs // 2, cx - cs // 2: cx + cs // 2]
    d = int(H * core_r * 2 * 1.15)
    crop = cv2.resize(crop, (d, d), interpolation=cv2.INTER_AREA)
    crop = grade_map(crop, AMBER, 1.15) * 1.05
    m = np.clip((core_r - R) / 0.018, 0, 1)[..., None]
    core = np.zeros_like(img)
    y0, x0 = H // 2 - d // 2, W // 2 - d // 2
    core[y0:y0 + d, x0:x0 + d] = crop
    img = img * (1 - m) + core * m
    ring = np.exp(-((R - core_r) / 0.006) ** 2)[..., None]
    img = img + glow * ring * (0.4 + 0.25 * kick)
    # soft feedback trails (MilkDrop memory), zooming outward
    if prev is not None:
        M = cv2.getRotationMatrix2D((W / 2, H / 2), 0.15, 1.018)
        fb = cv2.warpAffine(prev, M, (W, H), borderMode=cv2.BORDER_CONSTANT)
        img = 1 - (1 - np.clip(img, 0, 1)) * (1 - fb * (0.14 + 0.08 * kick))
    prev = np.clip(img, 0, 1)
    if MODE == 'clean':
        out = bloom(img, 0.7, 0.3 + 0.15 * kick, 14) * vig
    else:
        cell = 8
        dots = halftone_color(img, cell, 1.1 + 0.1 * kick, AMBER, 0.5)
        if MODE == 'halftone':
            out = dots
        else:  # hybrid
            # under-layer: soft, dimmed footage so forms still read between the dots
            under = cv2.GaussianBlur(img, (0, 0), 1.2) * 0.38
            hyb = 1 - (1 - np.clip(under, 0, 1)) * (1 - np.clip(dots, 0, 1))
            # radial blend: clean centre (couple + ring), dots take over towards the edges
            w_clean = np.clip(1 - (R - (core_r + 0.03)) / 0.28, 0, 1)[..., None]
            w_clean = w_clean * w_clean * (3 - 2 * w_clean)
            out = img * w_clean + hyb * (1 - w_clean)
        out = bloom(out, 0.7, 0.28 + 0.12 * kick, 10) * vig
    out = grain(tonemap(out), 0.01 if MODE == 'clean' else 0.006)
    wr.write(out)
wr.close()
