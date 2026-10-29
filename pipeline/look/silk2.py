"""Silk v2: liquid value field (smooth noise swept along a swirling flow), fine fur streaks as a light texture
(not relief), bright D-palette highlights up to yellow/white, ~2.5x faster motion, receding plane + depth of
field + the same sparse specks. Seamless loop of 32 beats at 146 BPM."""
import sys
import numpy as np, cv2, opensimplex
from common import W, H, luma, gradient_lut, Writer
import common
import silk as S                                  # reuse: palette, LIC noise, specks, plane, beat pulse
S.PERIOD = 46.0                                   # long, fine strands (shimmer pulses spaced further apart)
S.STEPS = 40
S.SHIM_CYCLES = 12

opensimplex.seed(36)
N, LOOP, FPS = S.N, S.LOOP, S.FPS
TW, TH = S.TW, S.TH
RAD = 1.45                                        # path radius in noise space: ~2.6x more change per loop than v1
LW, LH = 110, 74
lx = np.linspace(0, 2.3, LW); ly = np.linspace(0, 1.55, LH)
# value lattice with margin, sampled at flow-warped coordinates
MX = 0.9
vlx = np.linspace(-MX, 2.3 + MX, 260); vly = np.linspace(-MX, 1.55 + MX, 190)

# brighter D palette for the value field: more of the range in gold -> yellow -> white
PAL = gradient_lut([(0.00, S.hexrgb('16000A')), (0.12, S.hexrgb('5A0620')), (0.24, S.hexrgb('8A0A34')),
                    (0.36, S.hexrgb('C2183F')), (0.47, S.hexrgb('D8542C')), (0.58, S.hexrgb('E8A322')),
                    (0.70, S.hexrgb('F6B93A')), (0.82, S.hexrgb('FFDC96')), (0.92, S.hexrgb('FFF1D8')),
                    (1.00, S.hexrgb('FFFDF6'))])


def per(x, y, sc, off, ph):
    a = 2 * np.pi * ph
    return opensimplex.noise4array(x * sc + off, y * sc, np.array([RAD * np.cos(a) + off]),
                                   np.array([RAD * np.sin(a)]))[0, 0].astype(np.float32)


def fields(ph):
    psi = per(lx, ly, 1.0, 0.0, ph) + 0.45 * per(lx, ly, 2.1, 5.0, ph)
    dy, dx = np.gradient(psi)
    vx, vy = dy, -dx
    m = np.sqrt(vx * vx + vy * vy).mean() + 1e-6
    vx = (vx / m + 1.5).astype(np.float32); vy = (vy / m + 0.45).astype(np.float32)   # stronger current: fewer small whirlpools
    # liquid value: smooth noise sampled at coordinates swept back along the flow (2 passes) -> curling bands
    val = per(vlx, vly, 0.95, 11.0, ph) + 0.30 * per(vlx, vly, 2.0, 17.0, ph)
    X, Y = np.meshgrid(lx, ly)
    wx, wy = X.astype(np.float32), Y.astype(np.float32)
    for _ in range(3):
        mx = (wx - lx[0]) / (lx[1] - lx[0]); my = (wy - ly[0]) / (ly[1] - ly[0])
        u = cv2.remap(vx, mx.astype(np.float32), my.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        v = cv2.remap(vy, mx.astype(np.float32), my.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        wx = wx - u * 0.12; wy = wy - v * 0.12
    band = cv2.remap(val, ((wx - vlx[0]) / (vlx[1] - vlx[0])).astype(np.float32),
                     ((wy - vly[0]) / (vly[1] - vly[0])).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    n = np.sqrt(vx * vx + vy * vy) + 1e-6
    up = lambda a: cv2.resize(a, (TW, TH), interpolation=cv2.INTER_CUBIC)
    vxu, vyu = up(vx / n), up(vy / n); nn = np.sqrt(vxu * vxu + vyu * vyu) + 1e-6
    return vxu / nn, vyu / nn, up(band)


def render(i):
    ph = i / N
    t = ph * LOOP
    pulse = S.beat_pulse(t)
    vx, vy, band = fields(ph)
    st = S.lic(vx, vy, ph)
    st = st / (st.std() + 1e-6)
    # value composition: broad smooth darks and lights, balanced (centre the distribution, soft contrast curve)
    b = (band - np.median(band)) / (np.percentile(band, 95) - np.percentile(band, 5) + 1e-6)
    v = np.clip(0.5 + 1.15 * b, 0, 1)
    v = v * v * (3 - 2 * v)
    # fur: a light texture, strongest in the lights and along light/dark edges (like the reference)
    edge = np.clip(1 - np.abs(v - 0.55) / 0.45, 0, 1)
    fur = st * (0.035 + 0.06 * v + 0.05 * edge)
    lum = np.clip(v * 0.97 + fur + 0.02, 0, 1)
    col = PAL[(lum * 255).astype(np.uint8)]
    # soft sheen travelling along the streaks in the bright areas (shimmer), no relief shading
    sheen = np.clip(st - 1.2, 0, None) * np.clip((v - 0.55) / 0.45, 0, 1)
    col = col + PAL[255][None, None] * (sheen * (0.10 + 0.06 * pulse))[..., None]
    col = col * (0.95 + 0.10 * pulse)
    scr = cv2.warpPerspective(col, S.HM, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    far = cv2.GaussianBlur(scr, (0, 0), 4.0)
    near = cv2.GaussianBlur(scr, (0, 0), 1.8)
    d = S.DEPTH[..., None]
    sw = np.exp(-((d - 0.62) / 0.24) ** 2)
    img = scr * sw + (far * (d < 0.62) + near * (d >= 0.62)) * (1 - sw)
    img = img * (0.45 + 0.55 * np.clip(d * 1.5, 0, 1))
    img = S.specks(img, ph, pulse)
    hi = np.clip(img - 0.75, 0, None)
    img = img + cv2.resize(cv2.GaussianBlur(cv2.resize(hi, (W // 4, H // 4)), (0, 0), 4), (W, H)) * (0.5 + 0.25 * pulse)
    Lm = luma(img)[..., None]
    return np.clip(img * ((Lm / (1 + Lm * 0.35)) / np.maximum(Lm, 1e-5)) * 1.12, 0, 1)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'still':
        cv2.imwrite(sys.argv[3], (render(int(sys.argv[2])) * 255 + 0.5).astype(np.uint8))
    else:
        common.FPS = FPS
        wr = Writer('out/E_silk_loop_v2.mp4', 'out/stills_silk2', [0, 80, 160, 240, 328])
        for i in range(N):
            wr.write(render(i))
        wr.close()
