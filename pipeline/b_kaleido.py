"""Direction B - Ember Rosette: slow-turning kaleidoscope, leaves dissolving into the couple, ember gradient map."""
import sys
from common import *

DUR = float(sys.argv[1]) if len(sys.argv) > 1 else 10
leaves = Clip('src/h3.mp4')
couple = Clip('src/h4.mp4')

y, x = np.mgrid[0:H, 0:W].astype(np.float32)
X = (x - W / 2) / H
Y = (y - H / 2) / H
R = np.sqrt(X * X + Y * Y)
A = np.arctan2(Y, X)
vig = vignette(k=0.75)


def kaleido(src, n, rot, scale, cx, cy, ripple=0.0, t=0.0):
    seg = 2 * np.pi / n
    a = np.mod(A + rot, seg)
    a = np.minimum(a, seg - a)
    r = R * scale * (1 + ripple * np.sin(R * 18 - t * 6) * np.exp(-R * 2))
    sx = (cx + r * np.cos(a) * H).astype(np.float32)
    sy = (cy + r * np.sin(a) * H).astype(np.float32)
    return cv2.remap(src, sx, sy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


N = int(DUR * FPS)
wr = Writer('out/B_kaleido.mp4', 'out/stills_B', [40, 165, 270])
prev = None
rot = 0.0
for i in range(N):
    t = i / FPS
    kick, swell = beat(t)
    rot += (0.07 + 0.25 * kick ** 3) / FPS
    mix = np.clip((t - BEAT * 8) / (BEAT * 1.5), 0, 1)
    mix = mix * mix * (3 - 2 * mix)
    s = 0.95 * (1 - 0.05 * kick) * (1 + 0.15 * swell)
    out = 0
    if mix < 1:
        src = leaves.get(i).astype(np.float32) / 255
        cx = W * (0.35 + 0.1 * np.sin(t * 0.3))
        cy = H * (0.45 + 0.08 * np.cos(t * 0.23))
        k1 = kaleido(src, 8, rot, s, cx, cy, 0.03 * kick, t)
        out = out + k1 * (1 - mix)
    if mix > 0:
        src = couple.get(i).astype(np.float32) / 255
        k2 = kaleido(src, 6, -rot * 0.8, s * 0.55, W * 0.5, H * 0.6, 0.03 * kick, t)
        out = out + np.clip(k2 * 1.5, 0, 1) ** 0.85 * mix
    g = grade_map(out, EMBER, 1.2, 0.36 + 0.06 * (1 - mix))
    img = g * 0.8 + warm_split(out, 0.8) * 0.2
    if prev is not None:   # inward feedback: the rosette leaves warm echoes as it turns
        M = cv2.getRotationMatrix2D((W / 2, H / 2), -0.4, 0.985)
        fb = cv2.warpAffine(prev, M, (W, H))
        img = np.maximum(img, fb * (0.55 + 0.25 * kick))
    prev = img.copy()
    img = bloom(img, 0.65, 0.35 + 0.45 * kick, 16) * vig
    wr.write(grain(tonemap(img), 0.008))
wr.close()
