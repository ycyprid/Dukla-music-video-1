"""Leaves particles v2: motion comes only from the source clip (fixed framing, no camera move, no motes,
no feedback/trails, no other footage). Depth drives focus + light, not position. Pulse = light only.
Colour follows the locked A grade, with lifted shadows."""
import sys, glob
from common import *

DUR = float(sys.argv[1]) if len(sys.argv) > 1 else 10
DEP = sys.argv[2] if len(sys.argv) > 2 else 'depth/leaves_stab2'
src = Clip('src/h5.mp4', int(DUR * FPS) + 2)
dfiles = sorted(glob.glob(f'{DEP}/d*.png'))
N = min(int(DUR * FPS), len(dfiles), src.n)
rng = np.random.default_rng(70)

STEP = 3
gy, gx = np.mgrid[0:H:STEP, 0:W:STEP]
gx = gx.ravel().astype(np.float32); gy = gy.ravel().astype(np.float32)
sx = np.clip(gx + rng.uniform(-0.5, 0.5, gx.size).astype(np.float32) * STEP, 0, W - 1)
sy = np.clip(gy + rng.uniform(-0.5, 0.5, gy.size).astype(np.float32) * STEP, 0, H - 1)
ix, iy = sx.astype(np.int32), sy.astype(np.int32)
x0 = np.floor(sx).astype(np.int64); y0 = np.floor(sy).astype(np.int64)
fx = sx - x0; fy = sy - y0
TAPS = []
for dx, dy, ww in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)), (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
    xi, yi = x0 + dx, y0 + dy
    ok = (xi < W) & (yi < H)
    TAPS.append((np.where(ok, yi * W + xi, 0), (ww * ok).astype(np.float32)))
EXPO = STEP * STEP * 0.8
vig = vignette(k=0.5)


def splat(col, w):
    img = np.zeros((H * W, 3), np.float32)
    for idx, ww in TAPS:
        wt = ww * w
        for c in range(3):
            img[:, c] += np.bincount(idx, weights=wt * col[:, c], minlength=H * W)
    return img.reshape(H, W, 3) * EXPO


def grade_A(img):
    """the locked A grade (warm split + contrast), then lift the darker tones"""
    # greens -> amber of the same brightness, so the canopy stays in A's warm world
    gr = np.clip((img[..., 1] - img[..., 2]) / 0.10, 0, 1)[..., None]
    amber = luma(img)[..., None] * np.array([0.35, 0.85, 1.35], np.float32)
    img = img * (1 - 0.8 * gr) + amber * 0.8 * gr
    g = warm_split(img, 0.55)
    g = np.clip((g - 0.5) * 1.25 + 0.42, 0, 1) ** 1.15
    return np.clip(g, 0, 1) ** 0.72 * 1.08                        # shadow / low-mid lift


wr = Writer('out/P_leaves_v2.mp4', 'out/stills_P_leaves_v2', [60, 150, 255])
for i in range(N):
    t = i / FPS
    pulse = soft_pulse(t)
    f = src.get(i).astype(np.float32) / 255
    d = cv2.imread(dfiles[i], -1).astype(np.float32) / 65535
    dp = d[iy, ix]
    c = f[iy, ix]
    lum = luma(c)
    # sky -> black space (soft mask, no hard flicker at edges)
    blue = np.clip((c[:, 0] - c[:, 2] * 1.0) / 0.12, 0, 1)
    far = np.clip((0.16 - dp) / 0.08, 0, 1)
    keep = 1 - np.maximum(blue, far)
    col = np.clip(grade_A(c[None])[0], 0, 1.2)
    w = (0.5 + 0.8 * lum) * (0.6 + 0.4 * dp) * keep
    # depth of field from depth: near leaves crisp, deeper layers soft
    near = splat(col, w * dp ** 1.5)
    deep = splat(col, w * (1 - dp ** 1.5))
    img = near + cv2.GaussianBlur(deep, (0, 0), 2.2) * 1.15
    img = img * 0.8 + cv2.GaussianBlur(img, (0, 0), 1.4) * 0.6
    img = img * (1 + 0.06 * pulse)                                # pulse = light only
    img = bloom(img, 0.7, 0.3 + 0.1 * pulse, 14) * vig            # same bloom/vignette family as A
    wr.write(grain(tonemap(img), 0.01))
wr.close()
