"""'Pixel propagation' silk in palette D: fur-like streaks along a swirling flow (line-integral convolution),
shimmer travelling along the streaks, lit as 3D relief on a receding plane with depth of field, sparse
specks floating in depth. Seamless loop of 32 beats at 146 BPM (~13.15 s), everything periodic in time.
usage: python3 silk.py [still <frame> <out.png>]"""
import sys
import numpy as np, cv2, opensimplex
from common import W, H, luma, gradient_lut, Writer
import common

opensimplex.seed(36)
BPM = 146.0
LOOP = 32 * 60 / BPM                         # 13.15 s
FPS = 25
N = int(round(LOOP * FPS))                   # 329 frames, sampled at exactly LOOP/N spacing
TW, TH = 1700, 1150                          # texture (plane) space
LW, LH = 96, 64                              # flow-field resolution
R = 0.55                                     # radius of the circular path in noise space -> periodic


def hexrgb(h):
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


PAL = gradient_lut([(0.00, hexrgb('140006')), (0.14, hexrgb('5A0620')), (0.28, hexrgb('8A0A34')),
                    (0.42, hexrgb('C2183F')), (0.56, hexrgb('D8542C')), (0.70, hexrgb('E8A322')),
                    (0.84, hexrgb('F6B93A')), (0.94, hexrgb('FFDC96')), (1.00, hexrgb('FFF1D8'))])

lx = np.linspace(0, 2.3, LW); ly = np.linspace(0, 1.55, LH)          # bigger swirls


def periodic(sc, off, ph, x=lx, y=ly):
    a = 2 * np.pi * ph
    return opensimplex.noise4array(x * sc + off, y * sc, np.array([R * np.cos(a) + off]),
                                   np.array([R * np.sin(a)]))[0, 0].astype(np.float32)


def fields(ph):
    psi = periodic(1.0, 0.0, ph) + 0.45 * periodic(2.1, 5.0, ph)
    dy, dx = np.gradient(psi)
    vx, vy = dy, -dx                                              # curl -> swirling, divergence-free
    m = np.sqrt(vx * vx + vy * vy).mean() + 1e-6
    vx = vx / m + 1.15; vy = vy / m + 0.35                         # + a steady current: streams and waves, fewer closed rings
    n = np.sqrt(vx * vx + vy * vy) + 1e-6
    vx = cv2.resize(vx / n, (TW, TH), interpolation=cv2.INTER_CUBIC)
    vy = cv2.resize(vy / n, (TW, TH), interpolation=cv2.INTER_CUBIC)
    n = np.sqrt(vx * vx + vy * vy) + 1e-6
    band = periodic(0.9, 11.0, ph) + 0.35 * periodic(1.9, 17.0, ph)  # big light/dark bands of the fluid
    band = cv2.resize(band, (TW, TH), interpolation=cv2.INTER_CUBIC)
    return vx / n, vy / n, band


rng = np.random.default_rng(3)
NOISE = rng.random((TH, TW)).astype(np.float32)
NOISE = cv2.GaussianBlur(NOISE, (0, 0), 1.1)                    # softer, silkier fibres
NOISE = (NOISE - NOISE.mean()) / NOISE.std()
yy, xx = np.mgrid[0:TH, 0:TW].astype(np.float32)
STEPS, DS = 34, 2.0                                             # longer streaks
PERIOD = 18.0                                                     # shimmer wavelength along a streak (steps)
SHIM_CYCLES = 8                                                   # shimmer travels 8 wavelengths per loop -> periodic


def lic(vx, vy, ph):
    """animated LIC: noise integrated along streamlines with a moving periodic kernel -> streaks + shimmer"""
    acc = np.zeros((TH, TW), np.float32); wsum = 0.0
    shift = SHIM_CYCLES * ph * PERIOD
    for d in (1, -1):
        px, py = xx.copy(), yy.copy()
        for s in range(STEPS):
            k = s * d
            w = 0.5 + 0.5 * np.cos(2 * np.pi * (k + shift) / PERIOD)     # travelling bright pulses along streaks
            w *= np.exp(-(s / STEPS) ** 2 * 1.5)
            acc += w * cv2.remap(NOISE, px, py, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            wsum += w
            ux = cv2.remap(vx, px, py, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            uy = cv2.remap(vy, px, py, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            px += d * ux * DS; py += d * uy * DS
    return acc / wsum


def beat_pulse(t):
    """soft swell on each beat, stronger on the half-time (every other) beat; periodic over the loop"""
    b = 60 / BPM
    ph = (t % b) / b
    up = np.clip(ph / 0.15, 0, 1); up = up * up * (3 - 2 * up)
    env = up * np.exp(-max(ph - 0.15, 0) / 0.35) + (1 - up) * np.exp(-(ph + 0.85) / 0.35)
    return float(env * (1.0 if int(round(t / b)) % 2 == 0 else 0.55))


# receding plane: texture -> screen homography (top of the plane further away)
SRC = np.float32([[0, 0], [TW, 0], [TW, TH], [0, TH]])
DST = np.float32([[W * 0.16, -H * 0.18], [W * 0.84, -H * 0.18], [W * 1.32, H * 1.12], [-W * 0.32, H * 1.12]])
HM = cv2.getPerspectiveTransform(SRC, DST)
sy_ = np.linspace(-0.18, 1.12, 1)
# per-pixel screen depth (1 = near bottom, 0 = far top) for depth of field + fog
DEPTH = np.clip((np.arange(H, dtype=np.float32) / H + 0.18) / 1.3, 0, 1)[:, None].repeat(W, 1)


rs = np.random.default_rng(9)
NSPK = 70
SPX = rs.uniform(-1, 1, NSPK); SPY = rs.uniform(-0.6, 0.5, NSPK); SPZ = rs.uniform(0, 1, NSPK)
SPPH = rs.uniform(0, 1, NSPK); SPK = rs.integers(1, 3, NSPK)


def specks(img, ph, pulse):
    """sparse, delicate specks drifting toward camera through the space (wrap once per loop)"""
    z = np.mod(SPZ - ph, 1.0)                                    # 1 = far, 0 = at camera
    depth = 0.35 + 3.2 * z
    fade = np.clip(z / 0.12, 0, 1) * np.clip((1 - z) / 0.2, 0, 1)
    u = W / 2 + SPX * W * 0.6 / depth * 1.6; v = H * 0.42 + SPY * H / depth * 1.6
    tw = (0.5 + 0.5 * np.sin(2 * np.pi * (SPK * ph + SPPH))) ** 3
    layer = np.zeros((H, W, 3), np.float32)
    col = PAL[230] * (0.5 + 1.2 * tw[:, None]) * fade[:, None] * (1 + 0.3 * pulse)
    for i in range(NSPK):
        r = max(0.6, 1.6 / depth[i] ** 1.2)                     # tiny, a little larger when close
        if 0 <= u[i] < W and 0 <= v[i] < H:
            cv2.circle(layer, (int(u[i] * 4), int(v[i] * 4)), int(r * 4), col[i].tolist(), -1, cv2.LINE_AA, 2)
    layer = cv2.GaussianBlur(layer, (0, 0), 0.8) + cv2.GaussianBlur(layer, (0, 0), 4.0) * 0.6
    return img + layer


def render(i):
    ph = i / N                                                    # 0..1 over the loop
    t = ph * LOOP
    pulse = beat_pulse(t)
    vx, vy, band = fields(ph)
    s = lic(vx, vy, ph)                                           # streak texture (~N(0, small))
    s = s / (s.std() + 1e-6)
    # height field: big fluid bands + fine fur streaks
    b = np.clip(0.5 + 1.25 * band, 0, 1); b = b * b * (3 - 2 * b)          # strong light/dark fluid bands
    hgt = np.clip(b * (0.82 + 0.16 * s) + 0.03 * s, 0, 1)
    # 3D lighting: normals from the height field, warm key light from the upper left + specular sheen
    gy_, gx_ = np.gradient(cv2.GaussianBlur(hgt, (0, 0), 1.2))
    nx, ny, nz = -gx_ * 55, -gy_ * 55, np.ones_like(hgt)
    nl = np.sqrt(nx * nx + ny * ny + nz * nz); nx /= nl; ny /= nl; nz /= nl
    L = np.array([-0.45, -0.55, 0.70]); L /= np.linalg.norm(L)
    diff = np.clip(nx * L[0] + ny * L[1] + nz * L[2], 0, 1)
    Hh = L + np.array([0, 0, 1.0]); Hh /= np.linalg.norm(Hh)
    spec = np.clip(nx * Hh[0] + ny * Hh[1] + nz * Hh[2], 0, 1) ** 40
    lum = np.clip(hgt ** 0.9 * (0.30 + 0.95 * diff), 0, 1)
    col = PAL[(lum * 255).astype(np.uint8)]
    col = col + PAL[250][None, None] * (spec * (1.1 + 0.5 * pulse) * np.clip(hgt * 1.4, 0, 1))[..., None]
    col = col * (0.92 + 0.12 * pulse)
    # onto the receding plane, depth of field + fog toward the far edge
    scr = cv2.warpPerspective(col, HM, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    far = cv2.GaussianBlur(scr, (0, 0), 4.5)
    near = cv2.GaussianBlur(scr, (0, 0), 2.0)
    d = DEPTH[..., None]
    sharp_w = np.exp(-((d - 0.62) / 0.22) ** 2)                   # focus band in the middle distance
    img = scr * sharp_w + (far * (d < 0.62) + near * (d >= 0.62)) * (1 - sharp_w)
    img = img * (0.25 + 0.75 * np.clip(d * 1.6, 0, 1))             # far fades into darkness
    img = specks(img, ph, pulse)
    hi = np.clip(img - 0.7, 0, None)
    img = img + cv2.resize(cv2.GaussianBlur(cv2.resize(hi, (W // 4, H // 4)), (0, 0), 4), (W, H)) * (0.6 + 0.3 * pulse)
    Lm = luma(img)[..., None]
    return np.clip(img * ((Lm / (1 + Lm * 0.6)) / np.maximum(Lm, 1e-5)) * 1.2, 0, 1)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'still':
        cv2.imwrite(sys.argv[3], (render(int(sys.argv[2])) * 255 + 0.5).astype(np.uint8))
    else:
        common.FPS = FPS
        wr = Writer('out/E_silk_loop.mp4', 'out/stills_silk', [0, 80, 160, 240, 328])
        for i in range(N):
            wr.write(render(i))
        wr.close()
