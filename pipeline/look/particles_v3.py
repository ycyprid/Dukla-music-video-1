"""Leaves particles v3: form -> hold with push-in -> disperse to black, in true 3D.
Timing on A's grid: 100 BPM, bar = 2.4 s. Form bar 1, hold bars 2-3, disperse bar 4, black to 10 s.
Particles are unprojected from the footage with the stabilised depth, so at rest they sit exactly on
the source image (source motion only); the camera and transitions move through real depth."""
import sys, glob
from common import *

DUR = 10.0
BAR = BEAT * 4                      # 2.4 s
T_FORM_END, T_DISP, T_DISP_END = BAR, BAR * 3, BAR * 4
NAME = sys.argv[1] if len(sys.argv) > 1 else 'leaves'
SRCP = {'leaves': 'src/h5.mp4', 'roses': 'src/h6b.mp4'}[NAME]
DEPD = {'leaves': 'depth/leaves_stab2', 'roses': 'depth/roses_b_s2'}[NAME]
GAIN = {'leaves': 1.0, 'roses': 1.15}[NAME]          # match shot brightness across clips
src = Clip(SRCP, int(DUR * FPS) + 2)
dfiles = sorted(glob.glob(f'{DEPD}/d*.png'))
N = int(DUR * FPS)
rng = np.random.default_rng(70)

STEP = 4                                                   # spaced out a little (was 3)
gy, gx = np.mgrid[0:H:STEP, 0:W:STEP]
gx = gx.ravel().astype(np.float32); gy = gy.ravel().astype(np.float32)
P = gx.size
sx = np.clip(gx + rng.uniform(-0.5, 0.5, P).astype(np.float32) * STEP, 0, W - 1)
sy = np.clip(gy + rng.uniform(-0.5, 0.5, P).astype(np.float32) * STEP, 0, H - 1)
ix, iy = sx.astype(np.int32), sy.astype(np.int32)
seed = rng.random(P).astype(np.float32)
seed2 = rng.random(P).astype(np.float32)
Xp = (sx - W / 2) / H; Yp = (sy - H / 2) / H              # image-plane coords
CAM = 2.2                                                  # rest camera distance
DZ = 0.9                                                   # depth extent of the scene

# forming: each particle starts deep behind the scene, scattered wide, and flies in (staggered)
F_X0 = rng.normal(0, 1.1, P).astype(np.float32)
F_Y0 = rng.normal(0, 0.7, P).astype(np.float32)
F_Z0 = (2.5 + 3.5 * rng.random(P)).astype(np.float32)
F_START = 0.15 + seed * 1.05                               # arrival starts spread over ~1 s
F_LEN = 1.05 + 0.25 * seed2
# dispersal: released front-to-back, flying toward and past the camera, opening outward, lifted by wind
D_DIR = np.stack([rng.normal(0, 1, P), rng.normal(0, 1, P) - 0.6, -np.abs(rng.normal(1.6, 0.5, P))], 1).astype(np.float32)


def ease_out(x):
    x = np.clip(x, 0, 1); return 1 - (1 - x) ** 3


def ease_in(x):
    x = np.clip(x, 0, 1); return x * x * x


def grade_A(img):
    gr = np.clip((img[..., 1] - img[..., 2]) / 0.10, 0, 1)[..., None]
    amber = luma(img)[..., None] * np.array([0.35, 0.85, 1.35], np.float32)
    img = img * (1 - 0.8 * gr) + amber * 0.8 * gr
    g = warm_split(img, 0.55)
    g = np.clip((g - 0.5) * 1.25 + 0.42, 0, 1) ** 1.15
    return np.clip(g, 0, 1) ** 0.72 * 1.08


def splat(u, v, col, w):
    img = np.zeros((H * W, 3), np.float32)
    x0 = np.floor(u).astype(np.int64); y0 = np.floor(v).astype(np.int64)
    fx = (u - x0).astype(np.float32); fy = (v - y0).astype(np.float32)
    for dx, dy, ww in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)), (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
        xi, yi = x0 + dx, y0 + dy
        ok = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
        idx = yi[ok] * W + xi[ok]; wt = (ww * w)[ok]
        for c in range(3):
            img[:, c] += np.bincount(idx, weights=wt * col[ok, c], minlength=H * W)
    return img.reshape(H, W, 3)


vig = vignette(k=0.5)
EXPO = STEP * STEP * 0.62
wr = Writer(f'out/P_{NAME}_v3.mp4', f'out/stills_P_{NAME}_v3', [20, 45, 120, 230, 250, 270])
for i in range(N):
    t = i / FPS
    pulse = soft_pulse(t)
    f = src.get(i).astype(np.float32) / 255
    d = cv2.imread(dfiles[min(i, len(dfiles) - 1)], -1).astype(np.float32) / 65535
    dp = d[iy, ix]; c = f[iy, ix]; lum = luma(c)
    blue = np.clip((c[:, 0] - c[:, 2]) / 0.12, 0, 1)
    far = np.clip((0.16 - dp) / 0.08, 0, 1)
    keep = 1 - np.maximum(blue, far)
    col = np.clip(grade_A(np.clip(c * GAIN, 0, 1)[None])[0], 0, 1.2)
    w = (0.5 + 0.8 * lum) * (0.6 + 0.4 * dp) * keep

    # rest position: unproject so the rest camera sees exactly the source image
    Z = (1 - dp) * DZ
    X = Xp * (Z + CAM) / CAM; Y = Yp * (Z + CAM) / CAM

    # camera: settles in during forming, slow push-in through the hold, keeps pushing into the dispersal
    kp = np.clip((t - 0.6) / (T_DISP_END - 0.6), 0, 1)
    push = 0.20 * kp * kp * (3 - 2 * kp)                               # ~9% closer by the end, eased
    cam = CAM - push + 0.25 * (1 - ease_out(t / T_FORM_END))

    def pose(tt):
        """particle positions + visibility at time tt (for motion-blur sub-samples)"""
        k = ease_out((tt - F_START) / F_LEN)                         # 0 = start, 1 = in place
        swirl = (1 - k) * 1.2
        cs_, sn_ = np.cos(swirl), np.sin(swirl)
        fx0 = F_X0 * cs_ - F_Y0 * sn_; fy0 = F_X0 * sn_ + F_Y0 * cs_
        px = fx0 + (X - fx0) * k; py = fy0 + (Y - fy0) * k; pz = F_Z0 + (Z - F_Z0) * k
        vis = np.clip((tt - F_START) / 0.35, 0, 1)
        # dispersal, released by depth: near first
        rel = T_DISP + (1 - dp) * 0.9 + seed * 0.35
        a = ease_in((tt - rel) / 1.6) * 2.4
        px = px + D_DIR[:, 0] * a * 0.55 + (px * a * 0.5)
        py = py + D_DIR[:, 1] * a * 0.40 + (py * a * 0.5)
        pz = pz + D_DIR[:, 2] * a
        vis = vis * (1 - np.clip((tt - rel - 0.9) / 1.0, 0, 1))
        return px, py, pz, vis

    sub = 3 if (t < T_FORM_END + 0.3 or t > T_DISP) else 1          # motion blur only while flying
    layers = [np.zeros((H, W, 3), np.float32) for _ in range(3)]
    for s in range(sub):
        tt = t - s / (sub * FPS)
        px, py, pz, vis = pose(tt)
        zc = pz + cam
        front = zc > 0.12
        u = W / 2 + H * CAM * px / np.maximum(zc, 0.12)
        v = H / 2 + H * CAM * py / np.maximum(zc, 0.12)
        scale = CAM / np.maximum(zc, 0.12)                           # >1 = closer than rest
        ww = w * vis * front / sub
        # bokeh: particles nearer the camera become larger, softer discs
        b0 = scale < 1.5; b1 = (scale >= 1.5) & (scale < 3.0); b2 = scale >= 3.0
        for L, m, e in ((0, b0, 1.0), (1, b1, 1.0), (2, b2, 1.0)):
            if m.any():
                layers[L] += splat(u[m], v[m], col[m], (ww * np.minimum(scale, 6) ** 0.8)[m] * e)
    img = layers[0] + cv2.GaussianBlur(layers[1], (0, 0), 2.4) * 3.0 + cv2.GaussianBlur(layers[2], (0, 0), 5.5) * 9.0
    img = img * EXPO
    img = img * 0.9 + cv2.GaussianBlur(img, (0, 0), 1.0) * 0.35           # light softening, gaps stay open
    img = img * (1 + 0.06 * pulse)
    img = bloom(img, 0.7, 0.3 + 0.1 * pulse, 14) * vig
    wr.write(grain(tonemap(img), 0.01))
wr.close()
