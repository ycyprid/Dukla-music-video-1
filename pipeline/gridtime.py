"""Grid displacement (as the reference): the masked flower is cut into square cells; each cell shows its own copy
of the clip - slightly shifted and scaled, and delayed by a few frames. Delays/offsets vary smoothly across the
grid and drift over time, so motion ripples through the cells in waves. Flat (no 3D), our warm grade on black."""
import sys, glob
import cv2, numpy as np
import opensimplex
from common import W, H, luma, warm_split, bloom, vignette, Writer
import common

opensimplex.seed(11)
SRC = 'src/h6b.mp4'
DEP = sorted(glob.glob('depth/roses_b_s2/d*.png'))
FPS = 30
DUR = float(sys.argv[1]) if len(sys.argv) > 1 else 10.0
CELL = 36
MAXD = 28                  # max delay in frames
MAXOFF = 24.0
MAXZ = 0.16
ZOOM = 1.18
vig = vignette(k=0.45)


def grade(img):
    gr = np.clip((img[..., 1] - img[..., 2]) / 0.10, 0, 1)[..., None]
    amber = luma(img)[..., None] * np.array([0.35, 0.85, 1.35], np.float32)
    img = img * (1 - 0.7 * gr) + amber * 0.7 * gr
    g = warm_split(img, 0.45)
    g = np.clip((g - 0.5) * 1.2 + 0.47, 0, 1)
    return np.clip(g, 0, 1) ** 0.85 * 1.12


def mask_from_depth(d):
    m = np.clip((d - 0.50) / 0.12, 0, 1)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((11, 11), np.uint8))
    return cv2.GaussianBlur(m, (0, 0), 2.5)


# ---- load clip, masks; track the main flower so the frame follows the camera drift ----
cap = cv2.VideoCapture(SRC)
NF = min(int(DUR * FPS) + 2, len(DEP))
raw, masks, cents = [], [], []
for i in range(NF):
    ok, f = cap.read()
    d = cv2.imread(DEP[i], -1).astype(np.float32) / 65535
    m = mask_from_depth(d)
    big = (d > 0.62).astype(np.float32)
    ys, xs = np.nonzero(big)
    cents.append((xs.mean(), ys.mean()) if xs.size else (W / 2, H / 2))
    raw.append(f); masks.append(m)
cents = np.array(cents)
k = np.exp(-0.5 * (np.arange(-30, 31) / 12.0) ** 2); k /= k.sum()
sm = np.stack([np.convolve(np.pad(cents[:, a], 30, mode='edge'), k, mode='valid') for a in range(2)], 1)
FR = []
for i in range(NF):
    cx, cy = sm[i]
    cx = np.clip(cx, W / (2 * ZOOM), W - W / (2 * ZOOM)); cy = np.clip(cy, H / (2 * ZOOM), H - H / (2 * ZOOM))
    M = np.float32([[ZOOM, 0, W / 2 - ZOOM * cx], [0, ZOOM, H / 2 - ZOOM * cy]])   # crop always inside source
    img = grade(raw[i].astype(np.float32) / 255) * masks[i][..., None]
    FR.append((np.clip(cv2.warpAffine(img, M, (W, H)), 0, 1) * 255).astype(np.uint8))
del raw
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


def render(i):
    t = i / FPS
    env, dl, ox, oy, zs = fields(t)
    out = np.zeros((H, W, 3), np.float32)
    for r in range(GY):
        for c in range(GX):
            x0, y0 = OX + c * CELL, OY + r * CELL
            xa, ya, xb, yb = max(x0, 0), max(y0, 0), min(x0 + CELL, W), min(y0 + CELL, H)
            if xb <= xa or yb <= ya:
                continue
            e = env[r, c]
            delay = e * dl[r, c] * MAXD
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
            out[ya:yb, xa:xb] = pa / 255
    out = bloom(out * 1.15, 0.6, 0.4, 10) * vig
    Lm = luma(out)[..., None]
    return np.clip(out * ((Lm / (1 + Lm * 0.7)) / np.maximum(Lm, 1e-5)) * 1.15, 0, 1)


if __name__ == '__main__':
    common.FPS = FPS
    wr = Writer('out/G_gridtime.mp4', 'out/stills_gt', [60, 150, 240])
    for i in range(int(DUR * FPS)):
        wr.write(render(i))
    wr.close()
