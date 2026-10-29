from common import *
import os
os.makedirs('out/halftone_ABC', exist_ok=True)

def halftone_color(img, cell, gain=1.15, lift=0.0):
    h, w = img.shape[:2]
    gw, gh = w // cell, h // cell
    ccol = cv2.resize(img, (gw, gh), interpolation=cv2.INTER_AREA)       # per-cell colour
    cl = np.clip((luma(ccol) + lift) * gain, 0, 1)
    up_l = cv2.resize(cl, (gw * cell, gh * cell), interpolation=cv2.INTER_NEAREST)
    up_c = cv2.resize(ccol, (gw * cell, gh * cell), interpolation=cv2.INTER_NEAREST)
    py, px = (h - gh * cell) // 2, (w - gw * cell) // 2
    up_l = cv2.copyMakeBorder(up_l, py, h - gh * cell - py, px, w - gw * cell - px, cv2.BORDER_CONSTANT)
    up_c = cv2.copyMakeBorder(up_c, py, h - gh * cell - py, px, w - gw * cell - px, cv2.BORDER_CONSTANT)
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    fx = ((x - px) % cell) - cell / 2 + 0.5
    fy = ((y - py) % cell) - cell / 2 + 0.5
    d = np.sqrt(fx * fx + fy * fy)
    r = np.sqrt(up_l) * cell * 0.64
    mask = np.clip(r - d + 0.5, 0, 1)[..., None]
    # dot colour: frame colour, saturated slightly and normalised so small dots stay warm not brown
    col = up_c / (luma(up_c)[..., None] + 0.08) * (0.35 + 0.65 * up_l[..., None])
    ramp = AMBER[(up_l * 255).astype(np.uint8)]
    col = col * 0.45 + ramp * 0.55 * (0.6 + 0.6 * up_l[..., None])
    out = col * mask
    out = bloom(out, 0.7, 0.3, 6)
    return np.clip(tonemap(out), 0, 1)

picks = {'A_tunnel': 'out/stills_A/f0150.png', 'B_rosette': 'out/stills_B/f0040.png', 'C_wall': 'out/stills_C/f0260.png'}
row_f, row_c = [], []
for name, p in picks.items():
    img = cv2.imread(p).astype(np.float32) / 255
    for tag, cell in [('fine', 8), ('coarse', 14)]:
        o = halftone_color(img, cell)
        u8 = (o * 255 + 0.5).astype(np.uint8)
        cv2.imwrite(f'out/halftone_ABC/{name}_halftone_{tag}.png', u8)
        (row_f if tag == 'fine' else row_c).append(cv2.resize(u8, (640, 360), interpolation=cv2.INTER_AREA))
cv2.imwrite('out/halftone_ABC/board.jpg', np.vstack([np.hstack(row_f), np.hstack(row_c)]), [cv2.IMWRITE_JPEG_QUALITY, 92])
