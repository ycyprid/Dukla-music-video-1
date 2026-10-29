"""Direction C - Memory Wall: grid of time-delayed tiles that subdivides on downbeats; echoes ripple out from the centre."""
import sys, math
from common import *

DUR = float(sys.argv[1]) if len(sys.argv) > 1 else 10
couple = Clip('src/h4.mp4')
leaves = Clip('src/h1.mp4')
vig = vignette(k=0.5)
levels = [1, 2, 3, 5, 7]


def grid_n(t):
    bar = BEAT * 4
    k = t / bar
    j = int(k)
    f = min(1.0, (k - j) * bar / 0.7)          # 0.7s eased transition after each downbeat
    f = f * f * (3 - 2 * f)
    a = levels[min(j, len(levels) - 1)]
    b = levels[min(j + 1, len(levels) - 1)]
    prev = levels[max(0, min(j, len(levels) - 1) - 1)] if j > 0 else a
    return prev + (a - prev) * f if j > 0 else a


N = int(DUR * FPS)
wr = Writer('out/C_tiles.mp4', 'out/stills_C', [30, 140, 260])
gut = 3
for i in range(N):
    t = i / FPS
    kick, swell = beat(t)
    n = grid_n(t)
    tw, th = W / n, H / n
    canvas = np.zeros((H, W, 3), np.float32)
    c0 = math.floor(-n / 2 + 0.5) - 1
    cols = range(c0, -c0 + 1)
    for r in cols:
        for c in cols:
            x0 = W / 2 + (c - 0.5 + (0.5 if round(n) % 2 == 0 else 0) * 0) * tw
            # centre tile sits in the middle for odd n; grid symmetric
            x0 = W / 2 - tw / 2 + c * tw
            y0 = H / 2 - th / 2 + r * th
            x1, y1 = x0 + tw, y0 + th
            if x1 <= 0 or y1 <= 0 or x0 >= W or y0 >= H:
                continue
            dist = math.hypot(c, r)
            delay = int(round(dist * 6))
            src = leaves if (abs(c) + abs(r)) % 2 == 1 and n > 2.5 else couple
            fr = src.get(i - delay)
            ix0, iy0, ix1, iy1 = int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))
            w_, h_ = ix1 - ix0 - gut, iy1 - iy0 - gut
            if w_ < 4 or h_ < 4:
                continue
            tile = cv2.resize(fr, (w_, h_), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
            if c % 2:
                tile = tile[:, ::-1]
            # light wave travelling outward from the centre on each kick
            ph = ((t - dist * 0.06) % BEAT) / BEAT
            wave = math.exp(-ph * 6)
            fade = math.exp(-dist * 0.22)
            tile = tile * (0.55 + 0.45 * fade) * (0.9 + 0.35 * wave)
            gx0, gy0 = ix0 + gut // 2, iy0 + gut // 2
            sx0, sy0 = max(0, -gx0), max(0, -gy0)
            dx0, dy0 = max(0, gx0), max(0, gy0)
            dx1, dy1 = min(W, gx0 + w_), min(H, gy0 + h_)
            if dx1 <= dx0 or dy1 <= dy0:
                continue
            canvas[dy0:dy1, dx0:dx1] = tile[sy0:sy0 + dy1 - dy0, sx0:sx0 + dx1 - dx0]
    g = grade_map(canvas, AMBER, 1.15, 0.45)
    img = g * 0.75 + warm_split(canvas, 0.7) * 0.25
    # halation: red-orange glow bleeding from highlights (film memory)
    hal = bloom(np.zeros_like(img), 0, 0) if False else None
    hi = np.clip(luma(img) - 0.6, 0, None)[..., None] / 0.4
    small = cv2.resize(hi, (W // 4, H // 4))
    small = cv2.GaussianBlur(small, (0, 0), 3)
    hal = cv2.resize(small, (W, H))[..., None] * np.array([0.05, 0.25, 0.9], np.float32)
    img = img + hal * (0.5 + 0.5 * kick)
    img = bloom(img, 0.75, 0.25, 12) * vig
    wr.write(grain(tonemap(img), 0.018))
wr.close()
