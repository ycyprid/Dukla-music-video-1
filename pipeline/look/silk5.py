"""Liquid gradient (after the gradient-mesh reference) in our palette: long sweeping bands from the liquid warp field,
glowing rim lines at the band edges, soft satin lighting, film grain, sparse specks. Seamless 32-beat loop, 146 BPM."""
import sys
import numpy as np, cv2
from common import W, H, luma, gradient_lut, Writer
import common
import silk as S
import silk3 as L3

N, LOOP, FPS = S.N, S.LOOP, S.FPS
GWc, GHc = 360, 203
cx_ = np.linspace(0, 2.7, GWc).astype(np.float32); cy_ = np.linspace(0, 1.52, GHc).astype(np.float32)
CX, CY = np.meshgrid(cx_, cy_)
hx = S.hexrgb
PAL = gradient_lut([(0.00, hx('0C0005')), (0.20, hx('2A0412')), (0.36, hx('6A0A26')), (0.50, hx('B5183A')),
                    (0.62, hx('E0482A')), (0.74, hx('F08A2A')), (0.88, hx('F6B24E')), (1.00, hx('FDD995'))])
RIM = np.array(hx('FF5A3C')[::-1], np.float32) / 255
RIMC = np.array(hx('FFC48A')[::-1], np.float32) / 255
rng = np.random.default_rng(4)
GRAIN = [cv2.GaussianBlur(rng.standard_normal((H, W)).astype(np.float32), (0, 0), 0.6) for _ in range(8)]


def field(ph):
    # stretched coordinates -> long sweeping bands; two-level liquid domain warp (no vortices)
    X, Y = CX * 0.62, CY * 1.05
    a1, a2 = L3.lat(0.8, 0.0, ph), L3.lat(0.8, 7.0, ph)
    qx, qy = L3.samp(a1, X, Y), L3.samp(a2, X, Y)
    b1, b2 = L3.lat(0.9, 13.0, ph), L3.lat(0.9, 19.0, ph)
    rx = L3.samp(b1, X + 1.2 * qx, Y + 1.2 * qy); ry = L3.samp(b2, X + 1.2 * qx, Y + 1.2 * qy)
    v = L3.samp(L3.lat(0.7, 29.0, ph), X + 1.4 * rx, Y + 1.4 * ry)
    v2 = L3.samp(L3.lat(1.5, 37.0, ph), X * 1.1 + 0.8 * qx, Y * 1.1 + 0.8 * qy)     # gentle inner undulation
    up = lambda a: cv2.GaussianBlur(cv2.resize(a, (W, H), interpolation=cv2.INTER_CUBIC), (0, 0), 7.0)
    return up(v), up(v2)


def render(i):
    ph = i / N; t = ph * LOOP
    pulse = S.beat_pulse(t)
    v, v2 = field(ph)
    v = (v - np.median(v)) / (np.percentile(v, 96) - np.percentile(v, 4) + 1e-6)     # balanced composition
    v2 = v2 / (v2.std() + 1e-6)
    u = 0.46 + 0.80 * v + 0.07 * v2 * (0.4 + np.abs(v))                            # undulation inside the masses
    u = 1 / (1 + np.exp(-(u - 0.5) * 5.2))                                         # soft S-curve: no flat plateaus
    u = (u - 0.07) / 0.86
    u = np.clip(u, 0, 1)
    col = PAL[(u * 255).astype(np.uint8)]
    # satin lighting from the smooth value surface (soft key from upper left + broad sheen)
    gy, gx = np.gradient(cv2.GaussianBlur(u, (0, 0), 8.0))
    shade = np.clip(1 + (-gx * 0.7 - gy * 0.9) * 160, 0.75, 1.3)
    col = col * shade[..., None]
    # glowing rim lines where the bright bands meet the dark (thin core + soft glow)
    rimd = u - 0.40
    ribbon = np.exp(-(rimd / 0.06) ** 2)                                          # hot glowing ribbon at the edge
    core = np.exp(-(rimd / 0.022) ** 2)
    edge = np.clip(np.sqrt(gx * gx + gy * gy) * 700, 0, 1)
    col = col * (1 - 0.5 * ribbon[..., None] * edge[..., None]) + \
          (RIM[None, None] * ribbon[..., None] * 0.85 + RIMC[None, None] * core[..., None] * 0.35) * (edge * (0.85 + 0.3 * pulse))[..., None]
    hi = np.clip(col - 0.7, 0, None)
    col = col + cv2.resize(cv2.GaussianBlur(cv2.resize(hi, (W // 4, H // 4)), (0, 0), 5), (W, H)) * (0.55 + 0.25 * pulse)
    col = col * (0.95 + 0.08 * pulse)
    col = S.specks(col, ph, pulse)
    # film grain, stronger in the mids, cycling through a fixed set of frames (loop-safe)
    Lm = luma(np.clip(col, 0, 1))[..., None]
    col = np.clip(col + GRAIN[i % 8][..., None] * 0.035 * (0.35 + Lm), 0, None)
    Lm = luma(col)[..., None]
    return np.clip(col * ((Lm / (1 + Lm * 0.35)) / np.maximum(Lm, 1e-5)) * 1.12, 0, 1)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'still':
        cv2.imwrite(sys.argv[3], (render(int(sys.argv[2])) * 255 + 0.5).astype(np.uint8))
    else:
        common.FPS = FPS
        wr = Writer('out/E_gradient_loop.mp4', 'out/stills_grad', [0, 80, 160, 240, 328])
        for i in range(N):
            wr.write(render(i))
        wr.close()
