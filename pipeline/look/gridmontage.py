"""Grid-displacement montage: clips hand over to each other tile by tile. Every square is a shifted, zoomed,
time-offset copy of the footage (as the peony shot). Opening assembles from black, ending dissolves to black.
100 BPM at 25 fps: bar = 60 frames; each shot = 2 bars, handovers = 48 frames."""
import glob
import cv2, numpy as np
import opensimplex
from common import W, H, luma, bloom, Writer

opensimplex.seed(11)
CLIPS = [p for p in sorted(glob.glob('src/mont/m*.mp4')) if 'candle' not in p]
CELL = 36
MAXD, MAXOFF, MAXZ = 34, 40.0, 0.24
SLOT, TR, FADE = 120, 48, 5.0
NS = len(CLIPS)
TOTAL = NS * SLOT + TR                      # assemble-in at start, last shot dissolves at the end


def load(p):
    cap = cv2.VideoCapture(p); raw = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        raw.append(f.astype(np.float32) / 255)
    # per-clip level match: bring each clip's bright subject to a common level (low-key clips lifted)
    l = np.stack([luma(r) for r in raw[::20]])
    gain = float(np.clip(0.80 / max(np.percentile(l, 99.3), 1e-3), 0.8, 3.0))
    zoom = None
    if 'candle' in p:                                   # small subject: frame tighter on the flame
        acc = np.mean([cv2.GaussianBlur(luma(r), (0, 0), 15) for r in raw[::20]], 0)
        cy, cx = np.unravel_index(np.argmax(acc), acc.shape)
        zoom = (2.0, cx, cy + 90)
    fr = []
    for img in raw:
        if zoom:
            z, cx, cy = zoom
            cx = np.clip(cx, W / (2 * z), W - W / (2 * z)); cy = np.clip(cy, H / (2 * z), H - H / (2 * z))
            img = cv2.warpAffine(img, np.float32([[z, 0, W / 2 - z * cx], [0, z, H / 2 - z * cy]]), (W, H), flags=cv2.INTER_CUBIC)
        img = img * gain
        img = img / (1 + np.maximum(img - 0.85, 0) * 2)            # soft roll-off so lifted highlights don't clip
        m = np.clip((cv2.GaussianBlur(luma(img), (0, 0), 1.5) - 0.07) / 0.09, 0, 1)
        fr.append((np.clip(img * m[..., None], 0, 1) * 255).astype(np.uint8))
    print(p.split('/')[-1], 'gain %.2f' % gain, flush=True)
    return fr


FR = [load(p) for p in CLIPS]
print('loaded', [len(f) for f in FR], flush=True)

GX, GY = W // CELL + 1, H // CELL + 1
OX, OY = (W - GX * CELL) // 2, (H - GY * CELL) // 2
gx = np.arange(GX, dtype=np.float64); gy = np.arange(GY, dtype=np.float64)
yy, xx = np.mgrid[0:GY, 0:GX]
dc = np.hypot((xx - GX * 0.5) / GX, (yy - GY * 0.5) / GY)


def order(seed, centre_last=True):
    r = np.random.default_rng(seed).random((GY, GX))
    key = 0.65 * r + 0.35 * ((1 - dc / dc.max()) if centre_last else dc / dc.max())
    return np.argsort(np.argsort(key.ravel())).reshape(GY, GX) / (GX * GY - 1)


# handover k happens over frames [k*SLOT, k*SLOT+TR]: tiles switch from shot k-1 (or black) to shot k
# shot k plays from its handover start; the final handover (k = NS) goes to black
SWITCH = [k * SLOT + order(100 + k, centre_last=(k < NS)) ** 0.75 * (TR - FADE) for k in range(NS + 1)]


def fields(t):
    z = t * 0.22
    env = opensimplex.noise3array(gx * 0.16, gy * 0.16, np.array([z * 0.7 + 3]))[0]
    env = np.clip((env + 0.35) / 0.45, 0, 1); env = env * env * (3 - 2 * env)
    dl = opensimplex.noise3array(gx * 0.12, gy * 0.12, np.array([z]))[0] * 0.5 + 0.5
    ox = opensimplex.noise3array(gx * 0.25 + 20, gy * 0.25, np.array([z * 1.3]))[0]
    oy = opensimplex.noise3array(gx * 0.25 + 60, gy * 0.25, np.array([z * 1.3]))[0]
    zs = opensimplex.noise3array(gx * 0.2 + 90, gy * 0.2, np.array([z * 1.1]))[0]
    return env, dl, ox, oy, zs


def strength(g):
    # always strongly displaced; surges during each handover
    ph = (g % SLOT) / TR
    surge = np.sin(np.pi * np.clip(ph, 0, 1)) if ph <= 1 else 0.0
    return 0.85 + 0.4 * surge


def sample_tile(k, local, delay, s, dx, dy, xa, ya, xb, yb):
    fr = FR[k]; n = len(fr)
    j = local - delay
    j = float(np.clip(j, 0, n - 1.001))
    j0 = int(np.floor(j)); a = j - j0
    ccx, ccy = (xa + xb) / 2, (ya + yb) / 2
    X, Y = np.meshgrid(np.arange(xa, xb, dtype=np.float32), np.arange(ya, yb, dtype=np.float32))
    mx = (X / s + ccx + dx - ccx / s).astype(np.float32); my = (Y / s + ccy + dy - ccy / s).astype(np.float32)
    p = cv2.remap(fr[j0], mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT).astype(np.float32)
    if a > 0.01:
        p = p * (1 - a) + cv2.remap(fr[min(j0 + 1, n - 1)], mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT) * a
    return p / 255


def render(g):
    t = g / 25.0
    env, dl, ox, oy, zs = fields(t)
    env = np.clip(strength(g) * (0.6 + 0.4 * env), 0, 1.6)
    out = np.zeros((H, W, 3), np.float32)
    kin = min(g // SLOT, NS)                 # handover currently in progress / last started
    for r in range(GY):
        for c in range(GX):
            x0, y0 = OX + c * CELL, OY + r * CELL
            xa, ya, xb, yb = max(x0, 0), max(y0, 0), min(x0 + CELL, W), min(y0 + CELL, H)
            if xb <= xa or yb <= ya:
                continue
            # which shot this tile shows: newest handover whose switch time has begun for this tile
            k = -1; w_new = 1.0
            for kk in range(kin, -1, -1):
                st = SWITCH[kk][r, c]
                if g >= st:
                    k = kk; w_new = float(np.clip((g - st) / FADE, 0, 1)); w_new = w_new * w_new * (3 - 2 * w_new)
                    break
            e = env[r, c]
            delay = e * (dl[r, c] - 0.5) * MAXD * 1.4
            s = 1.0 + e * zs[r, c] * MAXZ
            dx, dy = e * ox[r, c] * MAXOFF, e * oy[r, c] * MAXOFF
            acc = 0
            for kk, wt in ((k, w_new), (k - 1, 1 - w_new)):
                if wt <= 1e-3 or kk < 0 or kk >= NS:
                    continue                 # -1 = black before the first shot, NS = black after the last
                local = g - kk * SLOT
                acc = acc + sample_tile(kk, local, delay, s, dx, dy, xa, ya, xb, yb) * wt
            if not isinstance(acc, int):
                out[ya:yb, xa:xb] = acc
    out = bloom(out, 0.8, 0.2, 10)
    return np.clip(out, 0, 1)


if __name__ == '__main__':
    import common; common.FPS = 25
    wr = Writer('out/G_montage.mp4', 'out/stills_mont', [30, 100, 140, 230, 260, 380, 500, 620, 640])
    for g in range(TOTAL):
        wr.write(render(g))
    wr.close()
