"""Cube <-> sphere dot form, v2 -- pristine version in the locked D look (matches the sheet benchmark's light).
- never stops: one full turn per loop about a tilted axis + a lagged, periodic speed swell and slow precession
- long, overlapping transitions: the morph travels across the form as a wave (2 s each way); every dot flies its own
  curved path (swells outward and swirls around the form's axis mid-flight), then settles
- light: small hot-white cores, gold glow, bloom over the bright areas, sparse flaring glints, focus on the front
  with only the back softening, fine background specks; delicate light-only pulse on the beat
Seamless loop of 14 beats at 146 BPM (144 frames @ 25 fps)."""
import sys
import numpy as np, cv2
import common
common.FPS = 25
from common import W, H, luma, Writer
import sheet_loop2 as L

FPS = 25
N = 144
T = N / FPS
BEATS = 14
rng = np.random.default_rng(21)

# ---- lattice: shell of a cube, 15 x 15 per face ----
n = 15
g = np.linspace(-1, 1, n)
X, Y, Z = np.meshgrid(g, g, g, indexing='ij')
sh = (np.abs(X) > 0.999) | (np.abs(Y) > 0.999) | (np.abs(Z) > 0.999)
P = np.stack([X[sh], Y[sh], Z[sh]], 1)
NP = len(P)
SPH = P / np.linalg.norm(P, axis=1, keepdims=True) * 1.15


def rodrigues(v, ax, ang):
    ax = ax / np.linalg.norm(ax)
    c, s = np.cos(ang)[:, None], np.sin(ang)[:, None]
    return v * c + np.cross(ax[None], v) * s + ax[None] * (v @ ax)[:, None] * (1 - c)


def rot_to(a, b):
    a = a / np.linalg.norm(a); b = b / np.linalg.norm(b)
    v = np.cross(a, b); c = np.dot(a, b)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K / (1 + c)


R0 = rot_to(np.array([1.0, 1.0, 1.0]), np.array([0.0, 0.0, 1.0]))     # corner-on at the loop start (hexagon)
AX = np.array([0.30, 1.0, 0.18]); AX /= np.linalg.norm(AX)           # main tumble axis (camera space)
AX2 = np.array([1.0, -0.15, 0.25]); AX2 /= np.linalg.norm(AX2)       # slow precession axis

# ---- overlapping motion ----
LAG = 0.05 + 0.13 * (P[:, 1] + 1) / 2 + rng.uniform(0, 0.04, NP)     # follow-through on the speed swells (s)
WIN = np.array([0.55, 0.62, 0.56]); WIN /= np.linalg.norm(WIN)        # direction the morph wave travels (cube space)
WOUT = np.array([-0.4, 0.75, -0.5]); WOUT /= np.linalg.norm(WOUT)
s_in = (P @ WIN - (P @ WIN).min()) / np.ptp(P @ WIN) + rng.uniform(-0.06, 0.06, NP)
s_out = (P @ WOUT - (P @ WOUT).min()) / np.ptp(P @ WOUT) + rng.uniform(-0.06, 0.06, NP)
DUR = 1.05 + rng.uniform(-0.12, 0.12, NP)                              # each dot's own flight time
SWIRL = rng.uniform(0.6, 1.0, NP)
IN0, IN_SPREAD = 0.55, 1.05                                            # wave in: starts 0.55 s, spreads over 1.05 s
OUT0, OUT_SPREAD = 3.25, 1.05


def ease(x):                                   # smootherstep with a whisper of overshoot -> soft settle
    x = np.clip(x, 0, 1)
    e = x * x * x * (x * (6 * x - 15) + 10)
    return e + 0.045 * np.sin(np.pi * x) ** 2 * np.sin(np.pi * np.clip((x - 0.5) * 2, 0, 1))


def morph_state(t):
    """per-dot morph amount m (0 cube .. 1 sphere) and mid-flight factor f (0 at rest, 1 mid-flight)"""
    t = np.mod(t, T)
    xi = (t - (IN0 + IN_SPREAD * s_in)) / DUR
    xo = (t - (OUT0 + OUT_SPREAD * s_out)) / DUR
    mi, mo = ease(xi), ease(xo)
    m = mi - mo
    f = np.sin(np.pi * np.clip(xi, 0, 1)) ** 1.5 + np.sin(np.pi * np.clip(xo, 0, 1)) ** 1.5
    return m, f, (xo > 0) * 1.0


def angles(t):
    """main turn (never stops) + lagged swell + precession, all periodic over the loop"""
    tl = t - LAG
    a1 = 2 * np.pi * t / T + 0.42 * np.sin(2 * np.pi * 2 * tl / T + 0.6)        # speed swells twice per loop, never < 0
    a2 = 0.22 * np.sin(2 * np.pi * tl / T + 1.1)
    return a1, a2


def positions(t):
    m, f, outphase = morph_state(t)
    base = P * (1 - m[:, None]) + SPH * m[:, None]
    # curved flight: swell outward + swirl around the form's own vertical axis, strongest mid-flight
    rad = base / np.maximum(np.linalg.norm(base, axis=1, keepdims=True), 1e-6)
    sw = np.cross(np.array([0.0, 1.0, 0.0])[None], base)
    direction = 1 - 2 * outphase                                                   # swirl one way in, the other way out
    base = base + rad * (0.20 * f)[:, None] + sw * (0.30 * f * SWIRL * direction)[:, None]
    v = base @ R0.T
    a1, a2 = angles(t)
    v = rodrigues(v, AX2, a2)
    v = rodrigues(v, AX, a1)
    return v, f


D = 7.4
FOC = H * 0.205 * D
LK = np.array([-0.40, 0.58, 0.71]); LK /= np.linalg.norm(LK)
GSEL = rng.random(NP) < 0.045; GK = rng.integers(1, 4, NP); GPH = rng.random(NP)
NSP = 360
SPX = rng.uniform(0, W, NSP); SPY = rng.uniform(0, H, NSP); SPA = rng.uniform(2, 10, NSP); SPP = rng.uniform(0, 6.28, NSP)
SPK = rng.integers(1, 3, NSP); SPB = rng.uniform(0.2, 1.0, NSP) ** 2
SPR = 0.6 + (rng.random(NSP) < 0.07) * rng.uniform(3, 8, NSP)
SPC = L.PAL[((0.55 + 0.35 * rng.random(NSP)) * 255).astype(np.uint8)]


def pulse(t):
    ph = np.mod(t / T * BEATS, 1.0)
    return np.exp(-ph / 0.28) * (1 - np.exp(-ph / 0.03))


def project(q):
    z = D - q[:, 2]
    return W / 2 + FOC * q[:, 0] / z, H / 2 - FOC * q[:, 1] / z, z


def splat_levels(U, V, C, Wt, Rr):
    lr = np.clip(np.log(Rr), L.LRAD[0], L.LRAD[-1])
    k0 = np.clip(np.searchsorted(L.LRAD, lr) - 1, 0, len(L.RAD) - 2)
    fr = ((lr - L.LRAD[k0]) / (L.LRAD[k0 + 1] - L.LRAD[k0])).astype(np.float32)
    img = np.zeros((H, W, 3), np.float32)
    for Lv, (kind, rad) in enumerate(L.LEVELS):
        wl = np.where(k0 == Lv, 1 - fr, 0) + np.where(k0 + 1 == Lv, fr, 0)
        mm = wl > 1e-4
        if not mm.any():
            continue
        ww = Wt[mm] * wl[mm]
        if kind == 'g':
            acc = np.stack([L.splat(U[mm], V[mm], ww * C[mm, c], H, W) for c in range(3)], -1)
            img += cv2.GaussianBlur(acc, (0, 0), rad) * (rad / 0.7) ** 1.2
        else:
            acc = np.stack([L.splat(U[mm] / 2, V[mm] / 2, ww * C[mm, c], H // 2, W // 2) for c in range(3)], -1)
            img += cv2.resize(cv2.filter2D(acc, -1, L.KER[Lv]) * (rad / 2.2) ** 1.2 * 1.6, (W, H))
    return img


def render(i):
    t = i / FPS
    p = pulse(t)
    q, f = positions(t)
    u, v, z = project(q)
    nrm = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-6)
    lam = np.clip(nrm @ LK, 0, 1)
    near = np.clip((q[:, 2] + 1.3) / 2.6, 0, 1)
    val = np.clip(0.02 + 0.50 * near ** 1.2 + 0.55 * lam * near, 0, 1)
    val = val * val * (3 - 2 * val)
    val = np.clip(val + 0.12 * f * near, 0, 1)                    # dots in flight catch a little more light
    col = L.PAL[((0.04 + 0.96 * val) * 255).astype(np.uint8)]
    fog = (1 - near)[:, None] ** 1.6 * 0.6
    col = col * (1 - fog) + L.DEEP * fog
    tw = GSEL * np.maximum(np.sin(2 * np.pi * (GK * t / T + GPH)), 0) ** 30 * (0.3 + 0.7 * val)
    w = (0.20 + 0.80 * val ** 1.3) * (0.10 + 0.90 * near ** 1.6) * 9.5 * (1 + 4.0 * tw)
    col = col * (1 - tw[:, None] * 0.6) + L.PAL[255][None] * tw[:, None] * 0.6
    zf = D - 1.05                                                  # focus on the front of the form
    r = np.maximum(0.75, 46.0 * np.maximum(1 / zf - 1 / z, 0) * D + 8.0 * np.maximum(1 / z - 1 / zf, 0) * D)
    # 180-degree shutter
    K = 6
    U, V, C, Wt, Rr = [], [], [], [], []
    for k in range(K):
        uk, vk, _ = project(positions(t + (k + 0.5) / K * 0.5 / FPS)[0])
        U.append(uk); V.append(vk); C.append(col); Wt.append(w / K); Rr.append(r)
    U, V, C, Wt, Rr = map(np.concatenate, (U, V, C, Wt, Rr))
    img = splat_levels(U, V, C, Wt, Rr)
    # background specks (periodic wobble / twinkle)
    ph = 2 * np.pi * SPK * t / T + SPP
    su = SPX + SPA * np.cos(ph); sv = SPY + SPA * 0.6 * np.sin(ph)
    stw = 0.35 + 0.65 * np.maximum(np.sin(ph * 2 + SPP), 0) ** 3
    img += splat_levels(su, sv, SPC, SPB * stw * 0.9, SPR)
    img *= L.EXP * (0.96 + 0.08 * p)
    # per-dot glow (bright dots + glints glow most) -- as the benchmark
    gw = w * (0.25 + 0.75 * val) ** 1.5 * (1 + 3.0 * tw)
    gacc = np.stack([L.splat(u / 2, v / 2, gw * col[:, c], H // 2, W // 2) for c in range(3)], -1)
    halo = cv2.GaussianBlur(gacc, (0, 0), 2.0) + cv2.GaussianBlur(gacc, (0, 0), 6.0) * 0.8
    img = img + cv2.resize(halo, (W, H)) * L.EXP * L.GLOW * (0.9 + 0.2 * p)
    # bloom over the bright areas + glint flares
    hi = np.clip(img - 0.55, 0, None)
    sm = cv2.resize(hi, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
    img = img + cv2.resize(cv2.GaussianBlur(sm, (0, 0), 3) * 0.35 + cv2.GaussianBlur(sm, (0, 0), 10) * 0.30, (W, H)) * (0.9 + 0.2 * p)
    fl = tw > 0.05
    if fl.any():
        facc = L.splat(u[fl], v[fl], tw[fl] * 6.0, H, W)
        flare = cv2.GaussianBlur(facc, (0, 0), 1.2) * 2.0 + cv2.GaussianBlur(facc, (0, 0), 5.0) * 1.2
        img = img + flare[..., None] * L.PAL[250][None, None]
    img = img * L.vig
    Lm = luma(img)[..., None]
    sat = 1.0 + 0.30 * np.exp(-((Lm - 0.35) / 0.25) ** 2)
    img = np.clip(Lm + (img - Lm) * sat, 0, None)
    Lm = luma(img)[..., None]
    Lt = Lm / (1 + Lm * 0.85)
    return np.clip(img * (Lt / np.maximum(Lm, 1e-5)) * 1.25, 0, 1)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'still':
        for fidx in sys.argv[2:]:
            cv2.imwrite(f'look/cube2_{int(fidx):03d}.png', (render(int(fidx)) * 255 + 0.5).astype(np.uint8))
    else:
        wr = Writer('out/D_cube_sphere_loop_v2_25fps.mp4', 'out/stills_cube2', [0, 25, 45, 70, 95, 115, 135])
        for i in range(N):
            wr.write(render(i))
        wr.close()
