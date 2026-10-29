"""Cube <-> sphere dot lattice (after the 'Pin on Horizon' reference) in the locked D halftone look.
A shell of dots on the faces of a cube, first seen corner-on (hexagon), tumbles one full turn about a tilted
axis, morphs into a sphere, holds, tumbles on and morphs back to the corner-on cube. Seamless loop of 14 beats
at 146 BPM (144 frames @ 25 fps).
Overlapping motion: every dot follows the rotation with its own small delay (height-based + random) and morphs
on its own staggered schedule with a springy overshoot, so the form twists and ripples in motion and settles
back crisp when it holds. Look: D palette, light from the front-left key + nearness, depth of field, per-dot
glow, glints, background specks, 180-degree motion blur, delicate light-only beat pulse."""
import sys, os
import numpy as np, cv2
import common
common.FPS = 25
from common import W, H, luma, Writer
import sheet_loop2 as L

FPS = 25
N = 144                                    # 14 beats @146 BPM = 5.753 s -> 144 frames
T = N / FPS
BEATS = 14
rng = np.random.default_rng(14)

# ---- the lattice: dots on the six faces of a cube (11 x 11 per face) ----
n = 11
g = np.linspace(-1, 1, n)
X, Y, Z = np.meshgrid(g, g, g, indexing='ij')
shell = (np.abs(X) > 0.999) | (np.abs(Y) > 0.999) | (np.abs(Z) > 0.999)
P = np.stack([X[shell], Y[shell], Z[shell]], 1)
NP = len(P)
Pinf = np.abs(P).max(1)
SPH = P / np.linalg.norm(P, axis=1, keepdims=True) * 1.12   # same dot on the sphere (radial mapping)


def rot_to(a, b):
    a = a / np.linalg.norm(a); b = b / np.linalg.norm(b)
    v = np.cross(a, b); c = np.dot(a, b)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K / (1 + c)


def axis_rot(ax, ang):
    ax = ax / np.linalg.norm(ax)
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K


R0 = rot_to(np.array([1.0, 1.0, 1.0]), np.array([0.0, 0.0, 1.0]))      # corner-on: body diagonal -> camera
R0 = axis_rot(np.array([0, 0, 1.0]), np.deg2rad(15)) @ R0              # a cube edge stands vertical-ish (pointy-top hexagon)
AX = np.array([0.32, 1.0, 0.22])                                       # tumble axis (camera space)


# ---- timeline (seconds, periodic over T) ----
def smooth(x):
    x = np.clip(x, 0, 1); return x * x * x * (x * (6 * x - 15) + 10)


tt = np.linspace(0, T, 4001)
speed = (np.exp(-((tt - 1.45) / 0.55) ** 2) + np.exp(-((tt - 4.25) / 0.55) ** 2)) + 0.06 * np.exp(-((tt - 2.85) / 0.6) ** 2)
speed *= np.clip((tt - 0.45) / 0.3, 0, 1) * np.clip((T - 0.45 - tt) / 0.3, 0, 1)   # holds at both ends (hexagon)
cum = np.concatenate([[0], np.cumsum((speed[1:] + speed[:-1]) / 2 * np.diff(tt))]); cum /= cum[-1]


def turn(t):                                   # rotation progress 0..1 (one full turn), periodic
    return np.interp(np.mod(t, T), tt, cum)


def back_out(x, s=1.9):                        # springy ease with overshoot
    x = np.clip(x, 0, 1); x1 = x - 1
    return 1 + (s + 1) * x1 ** 3 + s * x1 ** 2


def morph(t):                                  # 0 = cube, 1 = sphere ; in at ~1.7-2.5 s, out at ~3.6-4.4 s
    t = np.mod(t, T)
    a = back_out((t - 1.65) / 0.85)
    b = back_out((t - 3.55) / 0.85)
    return np.where(t < 1.65, 0.0, a) - np.where(t < 3.55, 0.0, b)


# per-dot overlap: rotation lag from height in the cube (bottom leads, top follows) + a little random
LAG_R = 0.07 + 0.11 * (P[:, 1] + 1) / 2 + rng.uniform(0, 0.04, NP)
# morph stagger: corners last (most distance to travel), with random jitter
LAG_M = 0.30 * (Pinf * 0 + np.linalg.norm(P, axis=1) - 1) / (np.sqrt(3) - 1) + rng.uniform(0, 0.16, NP)

# ---- look ----
D = 7.2                                        # camera distance (cube half-size = 1)
FOC = H * 0.215 * D                            # corner-on hexagon ~55 % of the frame height
LK = np.array([-0.45, 0.55, 0.70]); LK /= np.linalg.norm(LK)          # key light: front, upper left
SEL = rng.random(NP) < 0.05; GK = rng.integers(1, 4, NP); GPH = rng.random(NP)   # glints (integer cycles/loop)
NSP = 320
SPX = rng.uniform(0, W, NSP); SPY = rng.uniform(0, H, NSP); SPA = rng.uniform(2, 9, NSP); SPP = rng.uniform(0, 6.28, NSP)
SPK = rng.integers(1, 3, NSP); SPB = rng.uniform(0.2, 1.0, NSP) ** 2; SPR = 0.6 + (rng.random(NSP) < 0.07) * rng.uniform(3, 8, NSP)
SPC = L.PAL[((0.55 + 0.35 * rng.random(NSP)) * 255).astype(np.uint8)]


def pulse(t):                                  # delicate per-beat swell, periodic (14 beats per loop)
    ph = np.mod(t / T * BEATS, 1.0)
    return np.exp(-ph / 0.28) * (1 - np.exp(-ph / 0.03))


def positions(t):
    """3D dot positions (camera space) at time t, with per-dot overlap"""
    tr = t - LAG_R
    ang = 2 * np.pi * turn(tr)
    m = morph(t - LAG_M)[:, None]
    base = P * (1 - m) + SPH * m
    out = np.empty_like(base)
    # rotation per dot (each dot its own phase of the same tumble)
    ax = AX / np.linalg.norm(AX)
    c, s = np.cos(ang)[:, None], np.sin(ang)[:, None]
    v = base @ R0.T
    out = v * c + np.cross(ax[None], v) * s + ax[None] * (v @ ax)[:, None] * (1 - c)   # Rodrigues
    return out


def project(q):
    z = D - q[:, 2]
    return W / 2 + FOC * q[:, 0] / z, H / 2 - FOC * q[:, 1] / z, z


def render(i):
    t = i / FPS
    p = pulse(t)
    q = positions(t)
    u, v, z = project(q)
    nrm = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-6)
    lam = np.clip(nrm @ LK, 0, 1)
    near = np.clip((q[:, 2] + 1.25) / 2.5, 0, 1)                    # 0 = back of the form, 1 = front
    val = np.clip(0.06 + 0.52 * near + 0.42 * lam * (0.4 + 0.6 * near), 0, 1)
    val = val * val * (3 - 2 * val)
    col = L.PAL[((0.08 + 0.92 * val) * 255).astype(np.uint8)]
    fog = (1 - near)[:, None] ** 1.5 * 0.55
    col = col * (1 - fog) + L.DEEP * fog
    tw = SEL * np.maximum(np.sin(2 * np.pi * (GK * t / T + GPH)), 0) ** 24 * (0.4 + 0.6 * val)
    w = (0.30 + 0.70 * val) * (0.35 + 0.65 * near) * (1 + 2.0 * tw) * 6.5
    col = col * (1 - tw[:, None] * 0.5) + L.PAL[255][None] * tw[:, None] * 0.5
    zf = D - 0.55
    r = np.maximum(1.5, 70.0 * np.abs(1 / z - 1 / zf) * D)
    # 180-degree motion blur: sample the dots across the open shutter
    K = 7
    U, V, C, Wt, Rr = [], [], [], [], []
    for k in range(K):
        ts = t + (k + 0.5) / K * 0.5 / FPS
        uk, vk, _ = project(positions(ts))
        U.append(uk); V.append(vk); C.append(col); Wt.append(w / K); Rr.append(r)
    U, V, C, Wt, Rr = map(np.concatenate, (U, V, C, Wt, Rr))
    # background specks: sparse, small circular wobble + twinkle, periodic
    ph = 2 * np.pi * SPK * t / T + SPP
    su = SPX + SPA * np.cos(ph); sv = SPY + SPA * 0.6 * np.sin(ph)
    stw = 0.35 + 0.65 * np.maximum(np.sin(ph * 2 + SPP), 0) ** 3
    U = np.concatenate([U, su]); V = np.concatenate([V, sv]); C = np.concatenate([C, SPC])
    Wt = np.concatenate([Wt, SPB * stw * 0.9]); Rr = np.concatenate([Rr, SPR])
    # focus levels (smooth split between the two nearest blur radii)
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
    img *= L.EXP * (0.96 + 0.08 * p)
    # per-dot glow
    m = Wt > 0
    gacc = np.stack([L.splat(U / 2, V / 2, Wt * C[:, c], H // 2, W // 2) for c in range(3)], -1)
    halo = cv2.GaussianBlur(gacc, (0, 0), 2.0) + cv2.GaussianBlur(gacc, (0, 0), 7.0) * 0.8
    img = img + cv2.resize(halo, (W, H)) * L.EXP * L.GLOW * 0.55 * (0.9 + 0.2 * p)
    img = img * L.vig
    Lm = luma(img)[..., None]
    sat = 1.0 + 0.30 * np.exp(-((Lm - 0.35) / 0.25) ** 2)
    img = np.clip(Lm + (img - Lm) * sat, 0, None)
    Lm = luma(img)[..., None]
    Lt = Lm / (1 + Lm * 0.85)
    return np.clip(img * (Lt / np.maximum(Lm, 1e-5)) * 1.25, 0, 1)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'still':
        for f in sys.argv[2:]:
            cv2.imwrite(f'look/cube_{int(f):03d}.png', (render(int(f)) * 255 + 0.5).astype(np.uint8))
    else:
        wr = Writer('out/D_cube_sphere_loop_25fps.mp4', 'out/stills_cube', [0, 30, 45, 62, 80, 100, 118])
        for i in range(N):
            wr.write(render(i))
        wr.close()
