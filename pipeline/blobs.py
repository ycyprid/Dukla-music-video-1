"""Halftone blobs in the club palette: evolving simplex field drawn as a dot grid on black.
usage: python3 blobs.py standalone  |  python3 blobs.py context"""
import sys
import opensimplex
from common import *

opensimplex.seed(70)
CELL = 14
GW, GH = W // CELL + 1, H // CELL + 1
OX, OY = (W - (GW - 1) * CELL) / 2, (H - (GH - 1) * CELL) / 2   # centred grid


def hexrgb(h):
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


# club palette: burgundy -> plum -> dark gold -> gold -> pale gold core
PAL = gradient_lut([(0.00, hexrgb('3A0016')), (0.18, hexrgb('5D0024')), (0.38, hexrgb('6A1A31')),
                    (0.60, hexrgb('A6945A')), (0.80, hexrgb('C4AD71')), (1.00, hexrgb('F4ECD6'))])

gx = np.arange(GW, dtype=np.float64)
gy = np.arange(GH, dtype=np.float64)
F = 1.0 / 13.0           # noise frequency per dot -> ~3 blobs across the height


def field(t, pulse):
    z = t * 0.16
    wx = opensimplex.noise3array(gx * F * 0.6, gy * F * 0.6, np.array([z * 0.7 + 10]))[0]
    wy = opensimplex.noise3array(gx * F * 0.6, gy * F * 0.6, np.array([z * 0.7 + 40]))[0]
    # domain warp for soft, liquid shapes; slow drift
    xs = gx[None, :] * F + 0.9 * wx + t * 0.03
    ys = gy[:, None] * F + 0.9 * wy
    n1 = np.zeros((GH, GW))
    n2 = np.zeros((GH, GW))
    # evaluate on the warped coords (noise4 via loop-free trick: sample 3D at warped xy)
    for j in range(GH):
        n1[j] = [opensimplex.noise3(xs[j, i], ys[j, i], z) for i in range(GW)]
    n = n1
    lo = 0.02 - 0.05 * pulse
    v = np.clip((n - lo) / (0.62 - lo), 0, 1)
    v = v * v * (3 - 2 * v)
    return v.astype(np.float32)


# per-pixel helpers for dot drawing
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
ci = np.clip(np.round((xx - OX) / CELL), 0, GW - 1).astype(np.int32)
cj = np.clip(np.round((yy - OY) / CELL), 0, GH - 1).astype(np.int32)
dx = xx - (OX + ci * CELL)
dy = yy - (OY + cj * CELL)
dist = np.sqrt(dx * dx + dy * dy)
vig = vignette(k=0.35)


def draw_dots(v, pulse=0.0):
    vp = v[cj, ci]
    r = np.sqrt(vp) * CELL * 0.54
    mask = np.clip(r - dist + 0.6, 0, 1)
    col = PAL[(np.clip(vp, 0, 1) * 255).astype(np.uint8)]
    img = col * mask[..., None]
    img = bloom(img, 0.55, 0.35 + 0.1 * pulse, 10)
    return np.clip(tonemap(img * vig), 0, 1)


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'standalone'
    if mode == 'standalone':
        DUR = 12
        wr = Writer('out/D_blobs.mp4', 'out/stills_D', [30, 180, 330])
        for i in range(int(DUR * FPS)):
            t = i / FPS
            p = soft_pulse(t)
            wr.write(draw_dots(field(t, p), p))
        wr.close()
    else:
        # context cut: A tunnel -> dots emerge from the footage -> blobs -> back into A
        A = Clip('out/A_tunnel_clean.mp4', 480)
        DUR = 16
        T1, T2, T3, T4 = 4.0, 5.6, 10.4, 12.0
        wr = Writer('out/context_A_blobs_A.mp4', 'out/stills_ctx', [90, 135, 240, 345, 420])

        def ss(a, b, t):
            k = np.clip((t - a) / (b - a), 0, 1)
            return k * k * (3 - 2 * k)

        for i in range(int(DUR * FPS)):
            t = i / FPS
            p = soft_pulse(t)
            a_img = A.frames[min(i, A.n - 1)].astype(np.float32) / 255
            k = ss(T1, T2, t) * (1 - ss(T3, T4, t))           # 0 = footage, 1 = blobs
            if k <= 0:
                wr.write(a_img)
                continue
            vb = field(t, p)
            la = cv2.resize(luma(a_img), (GW, GH), interpolation=cv2.INTER_AREA)
            la = np.clip((la - 0.15) / 0.75, 0, 1)
            kv = k ** 0.8
            v = la * (1 - kv) + vb * kv                       # footage dots melt into the blob field
            dots = draw_dots(v, p)
            fade = np.clip(k * 1.8, 0, 1)                     # footage leaves faster than dots arrive
            fade = fade * fade * (3 - 2 * fade)
            wr.write(a_img * (1 - fade) + dots * fade)
        wr.close()
