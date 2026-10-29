"""Halftone blobs v2: faster, tempo-driven evolution; lifted + saturated palette; per-dot glow by brightness.
Tempo: 100 BPM grid shared with test A. Each beat gives the field a soft surge of change (no hard flicker)."""
import opensimplex
from common import *

opensimplex.seed(70)
CELL = 14
GW, GH = W // CELL + 1, H // CELL + 1
OX, OY = (W - (GW - 1) * CELL) / 2, (H - (GH - 1) * CELL) / 2


def hexrgb(h):
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


# PDF direction, lifted + saturated: deep burgundy -> bright burgundy -> crimson -> saturated gold -> pale gold
PAL = gradient_lut([(0.00, hexrgb('3C0014')), (0.14, hexrgb('8A0A34')), (0.30, hexrgb('C2183F')),
                    (0.48, hexrgb('D8542C')), (0.64, hexrgb('E8A322')), (0.80, hexrgb('F6B93A')),
                    (0.92, hexrgb('FFDC96')), (1.00, hexrgb('FFF1D8'))])

gx = np.arange(GW, dtype=np.float64)
gy = np.arange(GH, dtype=np.float64)
F = 1.0 / 13.0


def field(tau, pulse):
    z = tau * 0.42                                            # ~2.6x faster than v1
    wx = opensimplex.noise3array(gx * F * 0.6, gy * F * 0.6, np.array([z * 0.6 + 10]))[0]
    wy = opensimplex.noise3array(gx * F * 0.6, gy * F * 0.6, np.array([z * 0.6 + 40]))[0]
    xs = gx[None, :] * F + 0.9 * wx + tau * 0.06
    ys = gy[:, None] * F + 0.9 * wy
    n = np.array([[opensimplex.noise3(xs[j, i], ys[j, i], z) for i in range(GW)] for j in range(GH)])
    lo = 0.02 - 0.06 * pulse                                  # blobs swell gently on the beat
    v = np.clip((n - lo) / (0.62 - lo), 0, 1)
    return (v * v * (3 - 2 * v)).astype(np.float32)


yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
ci = np.clip(np.round((xx - OX) / CELL), 0, GW - 1).astype(np.int32)
cj = np.clip(np.round((yy - OY) / CELL), 0, GH - 1).astype(np.int32)
dist = np.sqrt((xx - (OX + ci * CELL)) ** 2 + (yy - (OY + cj * CELL)) ** 2)
vig = vignette(k=0.35)


def draw_dots(v, pulse=0.0):
    vp = v[cj, ci]
    r = np.sqrt(vp) * CELL * 0.54
    mask = np.clip(r - dist + 0.6, 0, 1)[..., None]
    col = PAL[(np.clip(vp, 0, 1) * 255).astype(np.uint8)]
    dots = col * mask
    # per-dot glow: each dot radiates by its own brightness (v^2.2), so cores glow and edge dots stay crisp
    g = dots * (vp[..., None] ** 2.2)
    small = cv2.resize(g, (W // 2, H // 2), interpolation=cv2.INTER_AREA)
    halo = cv2.GaussianBlur(small, (0, 0), 2.5) * 0.7 + cv2.GaussianBlur(small, (0, 0), 9) * 0.55
    halo = cv2.resize(halo, (W, H), interpolation=cv2.INTER_LINEAR)
    img = dots + halo * (0.85 + 0.25 * pulse)
    return np.clip(tonemap(img * vig), 0, 1)


if __name__ == '__main__':
    DUR = 10.0
    wr = Writer('out/D_blobs_v2.mp4', 'out/stills_D2', [30, 150, 270])
    tau = 0.0
    for i in range(int(DUR * FPS)):
        t = i / FPS
        p = soft_pulse(t)
        tau += (1.0 + 1.6 * p) / FPS                          # change surges with each beat, then settles
        wr.write(draw_dots(field(tau, p), p))
    wr.close()
