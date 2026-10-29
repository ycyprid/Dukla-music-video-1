"""Grid displacement (as the reference): the masked flower is cut into square cells; each cell shows its own copy
of the clip - slightly shifted and scaled, and delayed by a few frames. Delays/offsets vary smoothly across the
grid and drift over time, so motion ripples through the cells in waves. Flat (no 3D), our warm grade on black."""
import sys, glob
import cv2, numpy as np
import opensimplex
from common import W, H, luma, warm_split, bloom, vignette, Writer
import common

opensimplex.seed(11)
SRC = 'src/h8i.mp4'                       # retimed to a true 25 fps (motion-interpolated)
FPS = 25
MIX = 0.0
DUR = float(sys.argv[1]) if len(sys.argv) > 1 else 10.0
CELL = 36
MAXD = 34                  # max delay in frames
MAXOFF = 40.0
MAXZ = 0.24
ZOOM = 1.18
vig = vignette(k=0.45)


def hexrgb(h):
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


from common import gradient_lut
PAL_D = gradient_lut([(0.00, hexrgb('3C0014')), (0.14, hexrgb('8A0A34')), (0.30, hexrgb('C2183F')),
                      (0.48, hexrgb('D8542C')), (0.64, hexrgb('E8A322')), (0.80, hexrgb('F6B93A')),
                      (0.92, hexrgb('FFDC96')), (1.00, hexrgb('FFF1D8'))])     # test D palette


def grade(img):
    """map brightness onto test D's palette, then restore the original brightness exactly"""
    L0 = luma(img)
    col = PAL_D[(np.clip(L0, 0, 1) * 255).astype(np.uint8)]
    Lc = luma(col)
    col = col * (L0 / np.maximum(Lc, 1e-4))[..., None]
    # where scaling would clip (bright whites), blend toward the palette's cream so highlights stay clean
    return np.clip(col * MIX + img * (1 - MIX), 0, 1)                # mix palette D with the original colour


def mask_from_depth(d):
    m = np.clip((d - 0.50) / 0.12, 0, 1)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((11, 11), np.uint8))
    return cv2.GaussianBlur(m, (0, 0), 2.5)


# ---- load clip; mask by luminance (subject on black) ----
cap = cv2.VideoCapture(SRC)
FR = []
while len(FR) < int(DUR * FPS) + 2:   # (DUR from argv)
    ok, f = cap.read()
    if not ok:
        break
    img = f.astype(np.float32) / 255
    m = np.clip((cv2.GaussianBlur(luma(img), (0, 0), 1.5) - 0.06) / 0.08, 0, 1)
    FR.append((np.clip(grade(img) * m[..., None], 0, 1) * 255).astype(np.uint8))
NF = len(FR)
print('loaded', NF, flush=True)

GX, GY = W // CELL + 1, H // CELL + 1
OX, OY = (W - GX * CELL) // 2, (H - GY * CELL) // 2
gx = np.arange(GX, dtype=np.float64); gy = np.arange(GY, dtype=np.float64)


def fields(t):
    z = t * 0.22
    env = opensimplex.noise3array(gx * 0.16, gy * 0.16, np.array([z * 0.7 + 3]))[0]
    env = np.clip((env + 0.35) / 0.45, 0, 1); env = env * env * (3 - 2 * env)          # where the effect is active
    dl = opensimplex.noise3array(gx * 0.12, gy * 0.12, np.array([z]))[0] * 0.5 + 0.5    # smooth delay field -> waves
    ox = opensimplex.noise3array(gx * 0.25 + 20, gy * 0.25, np.array([z * 1.3]))[0]
    oy = opensimplex.noise3array(gx * 0.25 + 60, gy * 0.25, np.array([z * 1.3]))[0]
    zs = opensimplex.noise3array(gx * 0.2 + 90, gy * 0.2, np.array([z * 1.1]))[0]
    return env, dl, ox, oy, zs


def frame_at(j):
    j = int(np.clip(j, 0, NF - 1))
    return FR[j]


BAR = 60                                   # 100 BPM at 25 fps: one bar = 60 frames


def arc(i):
    """strongly displaced from the first frame, growing a little toward the end"""
    x = np.clip(i / 165, 0, 1)
    return 0.85 + 0.35 * x * x * (3 - 2 * x)


# dissolve: one tile at a time. Order = mostly random, gently biased so the edges go first and the
# heart of the flower last; tiles leave slowly at first, then more often (eased), each with a short clean fade.
_gw, _gh = W // CELL + 1, H // CELL + 1
_rng = np.random.default_rng(23)
_yy, _xx = np.mgrid[0:_gh, 0:_gw]
_dc = np.hypot((_xx - _gw * 0.42) / _gw, (_yy - _gh * 0.48) / _gh)
_key = 0.65 * _rng.random((_gh, _gw)) + 0.35 * (1 - _dc / _dc.max())
_rank = np.argsort(np.argsort(_key.ravel())).reshape(_gh, _gw) / (_gw * _gh - 1)
LEAVE = 92 + _rank ** 0.6 * 74                                  # frames 92..166, one at a time
FADE = 5.0


def render(i):
    t = i / FPS
    env, dl, ox, oy, zs = fields(t)
    S = arc(i)
    env = np.clip(S * (0.6 + 0.4 * env), 0, 1.6)                   # displaced everywhere, varying in strength
    vis = np.clip(1 - (i - LEAVE) / FADE, 0, 1)                     # each tile fades out on its own
    vis = vis * vis * (3 - 2 * vis)
    out = np.zeros((H, W, 3), np.float32)
    for r in range(GY):
        for c in range(GX):
            x0, y0 = OX + c * CELL, OY + r * CELL
            xa, ya, xb, yb = max(x0, 0), max(y0, 0), min(x0 + CELL, W), min(y0 + CELL, H)
            if xb <= xa or yb <= ya:
                continue
            e = env[r, c]
            if vis[r, c] <= 0:
                continue
            delay = e * (dl[r, c] - 0.5) * MAXD * 1.4                 # ahead or behind in time -> full effect from frame 1
            delay = float(np.clip(delay, i - (NF - 2), i))               # stay inside the clip
            fa = frame_at(i - int(np.floor(delay))); fb = frame_at(i - int(np.floor(delay)) - 1)
            a = delay - np.floor(delay)
            s = 1.0 + e * zs[r, c] * MAXZ
            dx, dy = e * ox[r, c] * MAXOFF, e * oy[r, c] * MAXOFF
            ccx, ccy = (xa + xb) / 2, (ya + yb) / 2
            # map output cell -> source: zoom about the cell centre + offset
            Mc = np.float32([[1 / s, 0, ccx - ccx / s - xa + dx * 0 + 0], [0, 1 / s, 0]])
            Mi = np.float32([[1 / s, 0, ccx + dx - (ccx) / s], [0, 1 / s, ccy + dy - (ccy) / s]])
            # sample region around the cell from the delayed frames
            X, Y = np.meshgrid(np.arange(xa, xb, dtype=np.float32), np.arange(ya, yb, dtype=np.float32))
            mx = Mi[0, 0] * X + Mi[0, 2]; my = Mi[1, 1] * Y + Mi[1, 2]
            pa = cv2.remap(fa, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT).astype(np.float32)
            if a > 0.01:
                pb = cv2.remap(fb, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT).astype(np.float32)
                pa = pa * (1 - a) + pb * a
            out[ya:yb, xa:xb] = pa / 255 * vis[r, c]
    out = bloom(out, 0.8, 0.2, 10)                                 # light glow only, no dimming
    return np.clip(out, 0, 1)


if __name__ == '__main__':
    common.FPS = FPS
    wr = Writer('out/G_peony_arc.mp4', 'out/stills_arc', [20, 70, 110, 130, 145, 160])
    for i in range(min(int(DUR * FPS), NF)):
        wr.write(render(i))
    wr.close()
