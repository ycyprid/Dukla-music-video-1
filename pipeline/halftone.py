from common import *

def halftone(l, cell=10, gain=1.0):
    """l: HxW luma 0..1 -> dot mask + per-cell luma (square grid, dot area ~ brightness)"""
    h, w = l.shape
    cl = cv2.resize(l, (w // cell, h // cell), interpolation=cv2.INTER_AREA)
    cl = np.clip(cl * gain, 0, 1)
    up = cv2.resize(cl, (w // cell * cell, h // cell * cell), interpolation=cv2.INTER_NEAREST)
    up = cv2.copyMakeBorder(up, 0, h - up.shape[0], 0, w - up.shape[1], cv2.BORDER_REPLICATE)
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    fx = (x % cell) - cell / 2 + 0.5
    fy = (y % cell) - cell / 2 + 0.5
    d = np.sqrt(fx * fx + fy * fy)
    r = np.sqrt(up) * cell * 0.62
    mask = np.clip(r - d + 0.5, 0, 1)
    return mask, up

WARM = gradient_lut([(0.0, (70, 8, 6)), (0.35, (190, 40, 16)), (0.65, (240, 140, 30)), (0.9, (255, 220, 120)), (1.0, (255, 250, 235))])

def colorize(mask, up):
    col = WARM[(up * 255).astype(np.uint8)]
    img = col * mask[..., None]
    return bloom(img, 0.75, 0.25, 8)

# 1) procedural blobs (metaballs)
y, x = np.mgrid[0:H, 0:W].astype(np.float32) / H
t = 3.0
f = np.zeros((H, W), np.float32)
rng = np.random.default_rng(3)
for k in range(9):
    cx = 0.9 + 0.7 * np.sin(t * (0.3 + 0.1 * k) + k * 1.7)
    cy = 0.5 + 0.38 * np.cos(t * (0.25 + 0.07 * k) + k * 2.3)
    s = 0.07 + 0.06 * rng.random()
    f += np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * s * s)) * (0.5 + 0.6 * rng.random())
f = np.clip(f, 0, 1.2) / 1.2
m, up = halftone(f ** 1.3, 12, 0.9)
cv2.imwrite('out/halftone_blobs.png', (np.clip(tonemap(colorize(m, up)), 0, 1) * 255).astype(np.uint8))

# 2) footage: couple + leaves
for name, path, fi, g in [('couple', 'src/h4.mp4', 60, 1.0), ('leaves', 'src/h1.mp4', 100, 0.75)]:
    c = Clip(path, fi + 1)
    img = c.get(fi).astype(np.float32) / 255
    l = luma(img)
    l = np.clip((l - 0.25) / 0.6, 0, 1) ** 1.1
    m, up = halftone(l, 11, g)
    cv2.imwrite(f'out/halftone_{name}.png', (np.clip(tonemap(colorize(m, up)), 0, 1) * 255).astype(np.uint8))
