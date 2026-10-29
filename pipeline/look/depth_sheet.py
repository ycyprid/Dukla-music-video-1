"""Style frames: the D dot-sheet look (sheet_loop3 benchmark) draped over a depth map.
A regular dot lattice is laid over the frame, pushed toward camera by depth (white = near), seen through
a slightly tilted perspective camera, so the rows bend over the relief. Value = height x key light x
facing (crests silver-white, faces gold, turning slopes red, ground deep burgundy -> black). Same palette,
focus levels, per-dot glow, twinkle glints, dust specks, vignette and tone curve as the benchmark.
usage: python3 depth_sheet.py depth.png out.png [seed]"""
import sys, os
import numpy as np, cv2
import sheet_loop2 as L
from common import W, H, luma, soft_pulse

PITCH_PX = float(__import__('os').environ.get('PITCH', '7.2'))                     # dot spacing on screen at the focus plane
REL = float(__import__('os').environ.get('REL', '0.55'))                         # relief depth (world units, frame height = 1)
Z0 = 2.4                            # camera distance to the background plane
TILT, YAW = float(__import__('os').environ.get('TILT', '0.16')), float(__import__('os').environ.get('YAW', '-0.10'))             # camera looks slightly down and from the side -> rows bend over the form
FOC = H * Z0 * 0.92
DOFK = float(os.environ.get('DOFK', '1'))        # depth-of-field strength
GAMMA = float(os.environ.get('GAMMA', '1'))      # <1 lifts mid depths (many subjects at different distances)
WIND = np.array([float(x) for x in os.environ.get('WIND', '0,0').split(',')])  # px/s; specks blown this way
RIMK = float(os.environ.get('RIMK', '1'))       # rim strength (lower for slim subjects)
WALL = float(os.environ.get('WALL', '0.50'))     # how far steep walls fall toward red                 # focal length so the plane fills the frame


RIMSCALE = {}
OUTS = float(os.environ.get('OUTS', '1'))        # output scale vs the W x H design (1.5 = 1920x1080, 3 = 3840x2160)
_DK = {}


def disc_k(Lv, S):
    """defocus disc for blur level Lv at output scale S (same look, more pixels)"""
    if S == 1:
        return L.KER[Lv]
    key = (Lv, S)
    if key not in _DK:
        _DK[key] = L.disc(L.LEVELS[Lv][1] * S)
    return _DK[key]
ORBIT = {}            # set per frame by the clip script: {'cx': centre x px, 'cy': centre y px}
TRANS = {}            # shot transitions (set by the clip script): see sheet_transition() / speck_fade()


def streak(u, v, c, w, r, vel, k, shutter=0.5 / 25, K=6):
    """motion streak along the wind: each speck smeared over the open shutter (K samples)"""
    sl = (np.arange(K) + 0.5) / K - 0.5
    U_ = (u[None] + vel[0] * k[None] * shutter * sl[:, None]).ravel()
    V_ = (v[None] + vel[1] * k[None] * shutter * sl[:, None]).ravel()
    return U_, V_, np.tile(c, (K, 1)), np.tile(w / K, K), np.tile(r, K)


def _gp(t):
    return t + 0.2 * 4.0 / (2 * np.pi) * (1 - np.cos(2 * np.pi * t / 4.0))


def speck_fade(t, th0, sp, w):
    """specks appear / vanish one at a time as a wave that travels around the swirl WITH them: each speck's turn
    comes when the wave front (sweeping from a gate on her left, in the travel direction) reaches it"""
    T_ = TRANS
    dirn = np.sign(w) if w != 0 else 1.0
    def wave(t0, spread):
        th_at = th0 + 2 * np.pi * w * sp * _gp(t0)
        d = np.mod(dirn * (th_at - (-np.pi / 2)), 2 * np.pi) / (2 * np.pi)   # 0 = at the gate, -> 1 = a full lap ahead
        return t0 + spread * d
    ti = wave(T_['sp_in0'], T_['sp_in_spread']); to = wave(T_['sp_out0'], T_['sp_out_spread'])
    fin = np.clip((t - ti) / T_['sp_fade'], 0, 1); fout = 1 - np.clip((t - to) / T_['sp_fade'], 0, 1)
    sm = lambda x: x * x * (3 - 2 * x)
    return sm(fin) * sm(fout)


def _sgnpow(x, p):
    return np.sign(x) * np.abs(x) ** p


def _back_out(x, s=0.8):                       # arrive with a whisper of overshoot, then settle
    x1 = np.clip(x, 0, 1) - 1
    return 1 + (s + 1) * x1 ** 3 + s * x1 ** 2


def sheet_transition(t, u, v, p_):
    """forming / dissolving of the dot sheet as a swirl around her vertical axis.
    Figure dots fly; every path starts / ends a set distance OFF-SCREEN (lower-left in, upper-right out), so the
    first and last frames are black. Faint background dots and the floor glow just fade with the formed fraction.
    Overlap: arrival in streams (low-frequency noise), torso first -> limbs follow through, own flight times, soft
    overshoot on arrival. Sparkles: trail sparks on the same paths + landing / take-off glints."""
    T_ = TRANS
    n = u.size
    rr = np.random.default_rng(777)                               # stable per-dot randomness (own stream)
    r1, r2, r3, r4, r5, r6 = (rr.random(n) for _ in range(6))
    cx, cy = ORBIT.get('cx', W / 2), ORBIT.get('cy', H / 2)
    dirn = T_['dir']
    mover = p_ > 0.35                                            # the figure flies; the empty field only fades
    # ---- stagger (overlapping motion) ----
    figd = np.hypot(u - cx, (v - cy) * 0.8)
    limb = np.clip(figd / (np.percentile(figd[mover], 95) + 1e-6 if mover.any() else 1), 0, 1)
    streams = 0.5 + 0.5 * np.sin(u / 83.0 + 1.7 * np.sin(v / 121.0)) * np.sin(v / 67.0 + 0.9)
    key = 0.26 * np.clip(1 - v / H, 0, 1) + 0.20 * np.clip(u / W, 0, 1) + 0.24 * limb + 0.18 * streams + 0.12 * r1
    sel = key[mover] if mover.any() else key
    key = (key - sel.min()) / (np.ptp(sel) + 1e-6)
    din = T_['in_dur'] * (0.8 + 0.4 * r2); dout = T_['out_dur'] * (0.8 + 0.4 * r5)
    tin = T_['in0'] + T_['in_spread'] * key; tout = T_['out0'] + T_['out_spread'] * key
    # ---- per-dot path ----
    x_rel = u - cx
    Rh = np.maximum(np.abs(x_rel) * 1.05, 30.0)
    th0 = np.arcsin(np.clip(x_rel / Rh, -1, 1))
    z0 = Rh * np.cos(th0)
    Dc = W * 1.6
    def swirl(sp, out):
        spc = np.clip(sp, 0, None)
        R = Rh + spc ** 1.1 * ((180 + 240 * r3) if out else (160 + 220 * r3))
        th = th0 + (dirn * sp * (1.5 + 1.5 * r4) if out else -dirn * sp * (1.4 + 1.4 * r4))
        z = R * np.cos(th); k = (Dc - z0) / (Dc - z)
        return cx + R * np.sin(th) * k, z
    xi1, _ = swirl(np.ones(n), False); xo1, _ = swirl(np.ones(n), True)
    DXi = -(xi1 + 150 + 260 * r6); DYi = (H + 150 + 200 * r2) - v           # -> fully off-screen, lower left
    DXo = (W + 150 + 260 * r6) - xo1; DYo = (-150 - 220 * r2) - v           # -> fully off-screen, upper right
    def state(tt):
        xi = (tt - tin) / din; xo = (tt - tout) / dout
        s_in = np.where(xi <= 0, 1.0, np.where(xi >= 1, 0.0, 1 - _back_out(xi)))
        s_out = np.clip(xo, 0, 1) ** 2.2
        return s_in, s_out
    def path(tt):
        s_in, s_out = state(tt)
        out = s_out > 0
        sp = np.where(out, s_out, s_in)
        xi_, zi = swirl(s_in, False); xo_, zo = swirl(s_out, True)
        xs = np.where(out, xo_, xi_); z = np.where(out, zo, zi)
        e = _sgnpow(sp, 1.6)
        x = xs + e * np.where(out, DXo, DXi)
        y = v + e * np.where(out, DYo, DYi)
        spc = np.clip(sp, 0, 1)
        y = y + 16 * np.sin(2 * np.pi * (1.6 * tt + r1)) * 4 * spc * (1 - spc)     # liquid wobble mid-flight only
        x = np.where(mover, x, u); y = np.where(mover, y, v); sp = np.where(mover, sp, 0.0)
        return x, y, sp, z
    K = 5
    smp = [path(t + (k + 0.5) / K * 0.5 / 25) for k in range(K)]
    x, y, sf, z = smp[K // 2]
    sfc = np.clip(sf, 0, 1)
    vis = np.clip((1 - sfc) / 0.06, 0, 1) * np.where(z < 0, 0.6, 1.0)   # lit the whole flight (they leave the frame)
    # formed fraction for the empty field + floor glow
    g_in = np.clip((t - (T_['in0'] + 0.55 * T_['in_spread'])) / (0.9 * T_['in_dur']), 0, 1)
    g_out = 1 - np.clip((t - T_['out0']) / (0.6 * T_['out_spread'] + 0.5 * T_['out_dur']), 0, 1)
    g = (g_in * g_in * (3 - 2 * g_in)) * (g_out * g_out * (3 - 2 * g_out))
    vis = np.where(mover, vis, g)
    # ---- sparkles ----
    SU, SV, SW, SR, SCi = [], [], [], [], []
    hosts = np.nonzero(mover)[0]
    if hosts.size:
        hr = np.random.default_rng(4242)
        for gi, lag in enumerate((0.05, 0.11, 0.18, 0.27)):
            pick = hosts[hr.random(hosts.size) < 0.05]
            if not pick.size:
                continue
            ang = hr.uniform(0, 2 * np.pi, n); sc_ = hr.uniform(12, 46, n); ph = hr.random(n); fq = hr.uniform(4, 9, n)
            for out, tt in ((False, t - lag), (True, t + lag * 0.6)):     # trail the arrival, lead the departure
                px, py, ps, _ = path(tt)
                s_in, s_out = state(tt)
                ss = (s_out if out else np.clip(s_in, 0, 1))[pick]
                alive = (ss > 0.04) & (ss < 0.96)
                if not alive.any():
                    continue
                q = pick[alive]; sq = ss[alive]
                env = np.sin(np.pi * sq) ** 0.8 * (0.35 + 0.65 * np.maximum(np.sin(2 * np.pi * (fq[q] * t + ph[q])), 0) ** 6)
                spread = sc_[q] * sq
                SU.append(px[q] + np.cos(ang[q]) * spread); SV.append(py[q] + np.sin(ang[q]) * spread)
                SW.append(env * (0.9 if out else 1.5)); SR.append(np.full(q.size, 0.7)); SCi.append(232 + (hr.random(q.size) * 23))
        # landing glints (forming) and take-off glints (dissolve): tiny flashes that drift up-right
        gl = hosts[np.random.default_rng(99).random(hosts.size) < 0.10]
        for tref, amp in ((tin[gl] + din[gl], 3.0), (tout[gl], 1.6)):
            tau = t - tref
            al = (tau > 0) & (tau < 0.6)
            if al.any():
                q = gl[al]; tq = tau[al]
                env = np.exp(-tq / 0.18) * (1 - np.exp(-tq / 0.025))
                SU.append(u[q] + dirn * 0.75 * 70 * tq); SV.append(v[q] - 0.55 * 70 * tq)
                SW.append(env * amp); SR.append(np.full(q.size, 0.65)); SCi.append(np.full(q.size, 250.0))
    if SU:
        sparks = (np.concatenate(SU), np.concatenate(SV), L.PAL[np.concatenate(SCi).astype(np.uint8)],
                  np.concatenate(SW), np.concatenate(SR))
    else:
        sparks = None
    return smp, vis, sf, sparks, g


def orbit_specks(rng, t, presence, U, V, C, Wt, Rr):
    """specks circling the subject's vertical axis (a slow vortex). Angular speed ORBIT_W rev/s (sign = direction:
    + sweeps left->right in FRONT of her). Front specks: bigger, softer, brighter; back specks hidden by her."""
    w = float(os.environ.get('ORBIT_W', '0.3'))
    cx, cy = ORBIT['cx'], ORBIT['cy']
    N = 700
    th0 = rng.uniform(0, 2 * np.pi, N)
    R = np.sqrt(rng.uniform(0.18, 1.0, N)) * W * 0.42             # orbit radius in px (at her distance)
    yy = rng.uniform(-0.62, 0.62, N) * H                            # height along her axis
    sp = (0.75 + 0.5 * rng.random(N)) * (420.0 / np.maximum(R, 120)) ** 0.5   # inner specks orbit a little faster
    gp = t + 0.2 * 4.0 / (2 * np.pi) * (1 - np.cos(2 * np.pi * t / 4.0))     # gentle gusts in the swirl
    th = th0 + 2 * np.pi * w * sp * gp
    D = W * 1.6                                                     # camera distance (px units) -> perspective
    zc = R * np.cos(th)                                             # + = in front of her
    k = D / (D - zc)
    u = cx + R * np.sin(th) * k
    v = cy + yy * k + zc * 0.06                                     # slight downward tilt: front specks a touch lower
    front = zc > 0
    occl = cv2.GaussianBlur(presence, (0, 0), 4)
    hid = cv2.remap(occl, u.astype(np.float32).reshape(1, -1), v.astype(np.float32).reshape(1, -1), cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)[0]
    vis = np.where(front, 1.0, 1 - np.clip(hid * 1.6, 0, 1))         # behind her -> hidden by the figure
    depthk = (zc / R.max() + 1) / 2                                 # 0 = far side, 1 = nearest
    tw = 0.45 + 0.55 * np.maximum(np.sin(rng.uniform(0, 6.28, N) + 2 * np.pi * rng.uniform(0.3, 0.8, N) * t), 0) ** 3
    col = L.PAL[((0.55 + 0.35 * rng.random(N)) * 255).astype(np.uint8)]
    wt = rng.uniform(0.25, 1.0, N) ** 2 * (0.5 + 1.6 * depthk) * tw * vis * 1.3 * float(os.environ.get('ORBIT_B', '3.5'))
    if TRANS:
        wt = wt * speck_fade(t, th0, sp, w)
    rr = 0.6 + 6.5 * np.clip(depthk - 0.55, 0, 1) ** 1.5 * (0.5 + rng.random(N))   # near ones out of focus
    # streak along the orbit (tangential velocity over the open shutter)
    vel = np.stack([R * np.cos(th) * k, -R * np.sin(th) * 0.06], 0) * 2 * np.pi * w * sp
    u, v, col, wt, rr = streak(u, v, col, wt, rr, vel, np.ones(N), shutter=0.22 / 25, K=4)
    return (np.concatenate([U, u]), np.concatenate([V, v]), np.concatenate([C, col]),
            np.concatenate([Wt, wt]), np.concatenate([Rr, rr]))


_FIT = {}


def _proj(gx, gy, d):
    X = (gx - W / 2) / H; Y = -(gy - H / 2) / H; Z = Z0 - REL * d
    ct, st, cy_, sy_ = np.cos(TILT), np.sin(TILT), np.cos(YAW), np.sin(YAW)
    Zc = Z - Z0 * 0.75; X1 = X * cy_ + Zc * sy_; Z1 = -X * sy_ + Zc * cy_
    Y2 = Y * ct - Z1 * st; Z2 = Y * st + Z1 * ct + Z0 * 0.75
    return W / 2 + FOC * X1 / Z2, H / 2 - FOC * Y2 / Z2


def fit_frame(u, v):
    """uniform zoom + shift so the footage's frame edges (at the subject's depth) land just outside the screen"""
    if not _FIT:
        d = float(os.environ.get('FIT_D', '0.8')); over = float(os.environ.get('FIT_OVER', '1.015'))
        xs = np.array([0, W, W / 2, W / 2], float); ys = np.array([H / 2, H / 2, 0, H], float)
        pu, pv = _proj(xs, ys, d)
        s_ = max(W / (pu[1] - pu[0]), H / (pv[3] - pv[2])) * over
        _FIT.update(cu=(pu[0] + pu[1]) / 2, cv=(pv[2] + pv[3]) / 2, s=s_)
    return W / 2 + (u - _FIT['cu']) * _FIT['s'], H / 2 + (v - _FIT['cv']) * _FIT['s']


def load_depth(p, lohi=None):
    d = cv2.imread(p, -1).astype(np.float32)
    d /= 65535.0 if d.max() > 255 else 255.0
    if d.ndim == 3:
        d = d[..., 0]
    d = cv2.resize(d, (W, H), interpolation=cv2.INTER_AREA)
    lo, hi = lohi if lohi is not None else (np.percentile(d, 2), np.percentile(d, 99.7))
    return np.clip((d - lo) / (hi - lo + 1e-6), 0, 1)


def lattice(levels=False):
    m = 0.10
    if not levels:
        xs = np.arange(-m * W, W * (1 + m), PITCH_PX); ys = np.arange(-m * H, H * (1 + m), PITCH_PX)
        gxs, gys = np.meshgrid(xs, ys)
        return gxs.ravel().astype(np.float32), gys.ravel().astype(np.float32)
    # half-pitch lattice with density levels: 0 = base grid, 1 = cell centres, 2 = the rest.
    # When the footage pushes in, levels 1 and 2 fade in so the dot density stays constant.
    xs = np.arange(-m * W, W * (1 + m), PITCH_PX / 2); ys = np.arange(-m * H, H * (1 + m), PITCH_PX / 2)
    I, J = np.meshgrid(np.arange(len(xs)), np.arange(len(ys)))
    gxs, gys = np.meshgrid(xs, ys)
    lv = np.where((I % 2 == 0) & (J % 2 == 0), 0, np.where((I % 2 == 1) & (J % 2 == 1), 1, 2)).ravel()
    return gxs.ravel().astype(np.float32), gys.ravel().astype(np.float32), lv, gxs.shape


def level_vis(lv, scale):
    ss = lambda a, b, x: np.clip((x - a) / (b - a), 0, 1) ** 2 * (3 - 2 * np.clip((x - a) / (b - a), 0, 1))
    return np.choose(lv, [np.ones_like(scale), ss(1.22, 1.45, scale), ss(1.72, 2.0, scale)]).astype(np.float32)


def local_scale(pts, shp):
    """how much the tracked grid has stretched around each dot (1 = original spacing)"""
    X = pts[0].reshape(shp); Y = pts[1].reshape(shp)
    xi, xj = np.gradient(X); yi, yj = np.gradient(Y)
    a = np.abs(xj * yi - xi * yj) / (PITCH_PX / 2) ** 2
    a = cv2.GaussianBlur(a.astype(np.float32), (0, 0), 3)
    return np.sqrt(np.maximum(a, 1e-4)).ravel()


def sampler(pts):
    gxs, gys = pts
    n = gxs.size; pad = (-n) % 1024
    mx = np.pad(gxs, (0, pad)).reshape(-1, 1024); my = np.pad(gys, (0, pad)).reshape(-1, 1024)
    if os.environ.get('EDGE') == 'black':      # subject on black: outside the footage is empty (no mirrored ghost)
        return lambda a: cv2.remap(a.astype(np.float32), mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0).ravel()[:n]
    return lambda a: cv2.remap(a.astype(np.float32), mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101).ravel()[:n]


def render(dep, seed=1, t=0.0, pts=None, vis=None, ids=None, state=None, lum=None):
    FL = float(os.environ.get('FLOOR', '0'))     # cut faint depth (smear trails) before lifting the mids
    if FL > 0:
        dep = np.clip((dep - FL) / (1 - FL), 0, 1)
    if GAMMA != 1:
        dep = dep ** GAMMA
    p = soft_pulse(t)
    rng = np.random.default_rng(seed)
    # lighting from the relief
    ds = cv2.GaussianBlur(dep, (0, 0), 2.5)
    gy, gx = np.gradient(ds)
    k = float(os.environ.get('NORMK', '55'))       # relief strength of the lighting (lower for small subjects)
    nx, ny, nz = -gx * k, -gy * k, np.ones_like(ds)
    nl = np.sqrt(nx * nx + ny * ny + 1); nx /= nl; ny /= nl; nz /= nl
    Lk = np.array([-0.42, -0.55, 0.72]); Lk /= np.linalg.norm(Lk)
    lam = np.clip(nx * Lk[0] + ny * Lk[1] + nz * Lk[2], 0, 1)
    Hh = Lk + np.array([0, 0, 1.0]); Hh /= np.linalg.norm(Hh)
    spec = np.clip(nx * Hh[0] + ny * Hh[1] + nz * Hh[2], 0, 1) ** 30
    fac = nz                                       # 1 facing camera, -> 0 on steep walls
    h = dep ** 1.4
    val = np.clip(h * (0.28 + 0.72 * lam ** 1.3) + 0.40 * spec * h, 0, 1)
    val = np.clip((val - 0.08) / 0.80, 0, 1); val = val * val * (3 - 2 * val)
    val = val * (1 - WALL + WALL * fac ** 3)                            # turning slopes go red-orange
    # silhouette rims: where the depth falls away, a glowing orange-red line (the benchmark's fold edges)
    eg = cv2.GaussianBlur(np.sqrt(gx * gx + gy * gy), (0, 0), 1.5)
    rs = RIMSCALE.setdefault('s', np.percentile(eg, 97) + 1e-6)          # fixed per clip (no flicker)
    rim = np.clip(eg / rs, 0, 1) ** 1.5 * np.clip(dep * 2.5, 0, 1)
    rim = rim * RIMK
    if lum is not None:
        # picture-driven value: the footage's own light (white dress -> silver, skin -> gold, shadow -> red/black);
        # depth adds only a little relief shading on top
        LUMK = float(os.environ.get('LUMK', '0.8'))
        lv = np.clip((lum - 0.04) / 0.86, 0, 1) ** float(os.environ.get('LUMG', '1.25'))   # contrast: shadows fall away, whites hit silver
        lv = lv * (0.85 + 0.30 * lam)                               # gentle relief from the depth normals
        val = np.clip(LUMK * lv + (1 - LUMK) * val, 0, 1)
        fac = np.ones_like(fac)                                     # brightness follows the picture, not the surface angle
    val = val * (1 - rim) + np.maximum(val, 0.42) * rim
    presence = np.clip((dep - 0.06) / 0.30, 0, 1) ** 1.3          # ground fades out to black
    if lum is not None:
        presence = np.clip((dep - 0.02) / 0.10, 0, 1)               # picture mode: inside the figure = fully present

    # lattice in world plane (with margin so the tilt never exposes an edge)
    if pts is None:
        pts = lattice()                                   # dots ride on the footage when pts are advected
    gxs, gys = pts
    samp = sampler(pts)
    d_, v_, p_, f_, e_ = samp(dep), samp(val), samp(presence), samp(fac), samp(rim)
    FAB = os.environ.get('FABRIC') and lum is not None
    if FAB:
        # where the footage has fine, bright, high-contrast detail (lace, satin highlights, tulle edges)
        lm_ = np.clip(lum, 0, 1.2).astype(np.float32)
        hp = np.abs(lm_ - cv2.GaussianBlur(lm_, (0, 0), 2.2))                    # fine local contrast
        fmap = np.clip(hp / float(os.environ.get('FAB_HP', '0.07')), 0, 1) ** 1.2 * np.clip((lm_ - 0.30) / 0.40, 0, 1)
        fd_ = samp(fmap)
    if state is not None:
        # per-dot temporal smoothing along the tracked dots (motion-correct: each dot follows its own surface).
        # Small changes (depth noise) are averaged; big changes (a limb arriving / leaving) pass straight through.
        cur = np.stack([d_, v_, p_, f_, e_], 0)
        key = ids if ids is not None else np.arange(cur.shape[1])
        if 'key' in state:
            order = np.argsort(state['key']); sk = state['key'][order]
            pos = np.clip(np.searchsorted(sk, key), 0, len(sk) - 1)
            hit = sk[pos] == key
            prev = state['val'][:, order[pos]]
            diff = np.abs(cur - prev).max(0)
            a = np.clip(0.30 + 0.70 * diff / 0.35, 0, 1) * hit + (1 - hit)
            cur = prev + (cur - prev) * a[None]
        state['key'] = key.copy(); state['val'] = cur.copy()
        d_, v_, p_, f_, e_ = cur
    X = (gxs - W / 2) / H; Y = -(gys - H / 2) / H; Z = Z0 - REL * d_
    # camera rotation (tilt about x, yaw about y)
    ct, st, cy_, sy_ = np.cos(TILT), np.sin(TILT), np.cos(YAW), np.sin(YAW)
    Zc = Z - Z0 * 0.75                                               # rotate around a point in the scene
    X1 = X * cy_ + Zc * sy_; Z1 = -X * sy_ + Zc * cy_
    Y2 = Y * ct - Z1 * st; Z2 = Y * st + Z1 * ct + Z0 * 0.75
    u = W / 2 + FOC * X1 / Z2 / (H / H); v = H / 2 - FOC * Y2 / Z2
    u = W / 2 + (u - W / 2) * 1.0; v = v
    if os.environ.get('FIT'):                     # fit the footage frame to the screen (+ overscan): no margin shows
        u, v = fit_frame(u, v)
    TRS = None
    if TRANS:
        TRS = sheet_transition(t, u, v, p_)

    col = L.PAL[((0.06 + 0.94 * v_) * 255).astype(np.uint8)]
    fog = np.clip(1 - p_ * 1.4, 0, 1)[:, None]
    col = col * (1 - fog * 0.6) + L.DEEP * fog * 0.6
    w = (0.25 + 0.75 * v_) * (0.10 + 0.90 * p_) * 3.4 * (0.35 + 0.65 * f_ ** 2) * float(os.environ.get("GAIN", "1"))
    # twinkle glints: sparse ground-anchored dots, strongest on bright parts
    if ids is None:
        sel = (rng.random(u.size) < 0.03) * rng.uniform(0.5, 1.0, u.size)
        ph = rng.random(u.size); kk = rng.uniform(0.35, 0.9, u.size)         # each glint: own phase / rate (Hz)
    else:                                                                    # glints tied to dot identity
        hh = lambda salt: L.hash01(ids, ids // 7 + 3, salt)
        sel = (hh(11) < 0.03) * (0.5 + 0.5 * hh(23)); ph = hh(37); kk = 0.35 + 0.55 * hh(41)
    tw = sel * np.maximum(np.sin(2 * np.pi * (kk * t + ph)), 0) ** 24 * (0.3 + 0.7 * v_) * p_
    if FAB:
        # fabric twinkles: a dot can only flash where real detail sits under it; own threshold + rhythm per dot
        fr_ = np.random.default_rng(31337)
        thr = fr_.uniform(0.15, 0.85, u.size); ph2 = fr_.random(u.size); kk2 = fr_.uniform(0.9, 2.4, u.size)
        gate = np.clip((fd_ - thr) / 0.18, 0, 1); gate = gate * gate * (3 - 2 * gate)
        tw = np.maximum(tw * 0.4, gate * np.maximum(np.sin(2 * np.pi * (kk2 * t + ph2)), 0) ** 20 * p_)
    w = w * (1 + 2.2 * tw)
    if vis is not None:
        w = w * vis
    col = col * (1 - tw[:, None] * 0.5) + L.PAL[255][None] * tw[:, None] * 0.5
    # depth of field: focus on the front of the flower
    zf = Z0 - REL * 0.82
    r = np.maximum(0.55, 26.0 * DOFK * np.abs(1 / Z2 - 1 / zf) * Z0)
    if TRS is not None:
        smp, tvis, tsf, SPK_, GFORM = TRS
        tsf = np.clip(tsf, 0, 1)
        w = w * tvis * (1 + 0.35 * tsf)                           # in flight: a touch brighter (liquid sparks)
        r = r + 3.5 * tsf ** 1.5                                  # and softer
        col = col * (1 - 0.35 * tsf[:, None]) + L.PAL[int(0.82 * 255)][None] * 0.35 * tsf[:, None]
        u, v = smp[len(smp) // 2][0], smp[len(smp) // 2][1]
    inf = (u > -60) & (u < W + 60) & (v > -60) & (v < H + 60)
    if TRS is not None:                                           # motion blur along each dot's flight
        live = inf & (w > 1e-4)
        K_ = len(smp)
        U = np.concatenate([sm_[0][live] for sm_ in smp]); V = np.concatenate([sm_[1][live] for sm_ in smp])
        C = np.tile(col[live], (K_, 1)); Wt = np.tile(w[live] / K_, K_); Rr = np.tile(r[live], K_)
        NSH = U.size; VAL_E = np.tile(v_[live], K_); RAW_R = Rr.copy()
        if SPK_ is not None:                                      # sparkle cores
            U = np.concatenate([U, SPK_[0]]); V = np.concatenate([V, SPK_[1]]); C = np.concatenate([C, SPK_[2]])
            Wt = np.concatenate([Wt, SPK_[3]]); Rr = np.concatenate([Rr, SPK_[4]])
    else:
        U, V, C, Wt, Rr = u[inf], v[inf], col[inf], w[inf], r[inf]
        NSH = U.size; VAL_E = v_[inf]
    GLX = None
    if FAB:                                                       # glint cores + flare (as on the roses, a touch brighter)
        gw = tw * float(os.environ.get('FAB_K', '6.0')) * (TRS[1] if TRS is not None else 1.0)
        gm = inf & (gw > 0.05)
        if gm.any():
            GLX = (u[gm], v[gm], gw[gm], np.repeat(L.PAL[250][None], gm.sum(), 0))
            U = np.concatenate([U, GLX[0]]); V = np.concatenate([V, GLX[1]]); C = np.concatenate([C, GLX[3]])
            Wt = np.concatenate([Wt, GLX[2]]); Rr = np.concatenate([Rr, np.full(gm.sum(), 0.6)])

    if ORBIT:
        U, V, C, Wt, Rr = orbit_specks(rng, t, presence, U, V, C, Wt, Rr)
    else:
        # dust specks (camera-space, defocused, faint gold)
        ND = 110
        du0 = rng.uniform(0, W, ND); dv0 = rng.uniform(0, H, ND)
        dr = rng.uniform(0.8, 9.0, ND)
        vx = rng.uniform(-10, 10, ND); vy = rng.uniform(-14, -3, ND) * (0.4 + dr / 9)   # slow drift, bigger = nearer = faster
        wind = np.hypot(*WIND) > 0
        if wind:
            # gusts: speed breathes 1 +- 25 % over ~3 s; position = integral of the speed (smooth, no jumps)
            gp = t + 0.25 * 3.0 / (2 * np.pi) * (1 - np.cos(2 * np.pi * t / 3.0))
            gs = 1 + 0.25 * np.sin(2 * np.pi * t / 3.0)
            dk = 0.55 + 0.9 * dr / 9                                     # nearer (bigger) specks fly faster: parallax
            du = np.mod(du0 + vx * 0.3 * t + WIND[0] * dk * gp, W); dv = np.mod(dv0 + vy * 0.3 * t + WIND[1] * dk * gp, H)
        else:
            du = np.mod(du0 + vx * t, W); dv = np.mod(dv0 + vy * t, H)
        dcol = np.repeat(L.PAL[int(0.80 * 255)][None], ND, 0)
        dw = rng.uniform(0.15, 0.55, ND) * 0.55 * (0.6 + 0.4 * np.sin(rng.uniform(0, 6.28, ND) + 1.3 * t) ** 2)
        if wind:
            du, dv, dcol, dw, dr = streak(du, dv, dcol, dw, dr, WIND * gs, dk)
        U = np.concatenate([U, du]); V = np.concatenate([V, dv]); C = np.concatenate([C, dcol])
        Wt = np.concatenate([Wt, dw]); Rr = np.concatenate([Rr, dr])
        # background specks: sparse fine points far behind the flower, slow drift, gentle twinkle
        NB = 420
        bu0 = rng.uniform(0, W, NB); bv0 = rng.uniform(0, H, NB)
        bz = rng.uniform(0.2, 1.0, NB)                                   # 1 = nearest of the far layer
        bjx = rng.uniform(-4, 4, NB); bjy = rng.uniform(1, 5, NB)
        if wind:
            bk_ = 0.25 + 0.75 * bz                                       # far layer slower than the near dust
            bu = np.mod(bu0 + bjx * bz * t + WIND[0] * bk_ * gp, W); bv = np.mod(bv0 - bjy * bz * t + WIND[1] * bk_ * gp, H)
        else:
            bu = np.mod(bu0 + bjx * bz * t, W); bv = np.mod(bv0 - bjy * bz * t, H)
        bph = rng.uniform(0, 6.28, NB); bk = rng.uniform(0.3, 0.8, NB)
        btw = 0.35 + 0.65 * np.maximum(np.sin(bph + 2 * np.pi * bk * t), 0) ** 3
        occl = cv2.GaussianBlur(presence, (0, 0), 6)
        bvis = 1 - np.clip(cv2.remap(occl, bu.astype(np.float32).reshape(1, -1), bv.astype(np.float32).reshape(1, -1), cv2.INTER_LINEAR)[0] * 1.6, 0, 1)
        bci = (0.55 + 0.35 * rng.random(NB)) * 255
        bcol = L.PAL[bci.astype(np.uint8)]
        bw = rng.uniform(0.25, 1.0, NB) ** 2 * 2.3 * btw * bvis
        br = 0.6 + 1.6 * (1 - bz) * rng.random(NB) + (rng.random(NB) < 0.06) * rng.uniform(3, 8, NB)   # a few soft bokeh                     # mostly crisp, a few soft
        if wind:
            bu, bv, bcol, bw, br = streak(bu, bv, bcol, bw, br, WIND * gs, bk_)
        U = np.concatenate([U, bu]); V = np.concatenate([V, bv]); C = np.concatenate([C, bcol])
        Wt = np.concatenate([Wt, bw]); Rr = np.concatenate([Rr, br])

    # ---- rasterise at the output scale (design stays at W x H; OUTS=1.5 -> 1920x1080, 3 -> 3840x2160) ----
    S = OUTS
    OW, OH = int(round(W * S)), int(round(H * S))
    E = S * S                                                      # same energy per dot spread over S^2 more pixels
    lr = np.clip(np.log(Rr), L.LRAD[0], L.LRAD[-1])
    k0 = np.clip(np.searchsorted(L.LRAD, lr) - 1, 0, len(L.RAD) - 2)
    fr = ((lr - L.LRAD[k0]) / (L.LRAD[k0 + 1] - L.LRAD[k0])).astype(np.float32)
    img = np.zeros((OH, OW, 3), np.float32)
    HTK = float(os.environ.get('HTK', '0'))
    if HTK > 0:
        # halftone dot size: in-focus sheet dots become sharp discs whose AREA follows the tone (uniform ink)
        R0 = float(os.environ.get('HTR0', '0.75')); FLAT = float(os.environ.get('HTFLAT', '0.5'))
        ht = np.zeros(U.size, bool); ht[:NSH] = Rr[:NSH] < 1.2
        val_ = np.zeros(U.size); val_[:NSH] = VAL_E
        Rd = R0 + HTK * np.clip(val_, 0, 1) ** 0.8                    # design-px radius
        Rref = R0 + HTK * 0.5 ** 0.8
        flat = (0.55 + 0.45 * val_) / (0.25 + 0.75 * val_)            # let size, not brightness, carry the tone
        wmul = (Rd / Rref) ** 2 * (flat ** FLAT) * float(os.environ.get('HTE', '1.0'))
        bins = np.round(Rd[ht] / 0.15).astype(int)
        hidx = np.nonzero(ht)[0]
        for b_ in np.unique(bins):
            q = hidx[bins == b_]
            rad_px = b_ * 0.15 * S
            ker = L.disc(max(rad_px, 0.5))
            acc = np.stack([L.splat(U[q] * S, V[q] * S, Wt[q] * wmul[q] * C[q, c], OH, OW) for c in range(3)], -1)
            img += cv2.filter2D(acc, -1, ker) * E * 0.95
        Wt = np.where(ht, 0.0, Wt)                                     # those dots are done
    for Lv, (kind, rad) in enumerate(L.LEVELS):
        wl = np.where(k0 == Lv, 1 - fr, 0) + np.where(k0 + 1 == Lv, fr, 0)
        mm = wl > 1e-4
        if not mm.any():
            continue
        ww = Wt[mm] * wl[mm]
        if kind == 'g':
            acc = np.stack([L.splat(U[mm] * S, V[mm] * S, ww * C[mm, c], OH, OW) for c in range(3)], -1)
            img += cv2.GaussianBlur(acc, (0, 0), rad * S) * (rad / 0.7) ** 1.2 * E
        else:
            acc = np.stack([L.splat(U[mm] * S / 2, V[mm] * S / 2, ww * C[mm, c], OH // 2, OW // 2) for c in range(3)], -1)
            img += cv2.resize(cv2.filter2D(acc, -1, disc_k(Lv, S)) * (rad / 2.2) ** 1.2 * 1.6 * E, (OW, OH))
    img *= L.EXP * (0.96 + 0.08 * p)
    # per-dot glow
    ui, vi, ci = u[inf], v[inf], col[inf]
    gw_ = w[inf] * (0.25 + 0.75 * v_[inf]) ** 1.5
    gacc = np.stack([L.splat(ui * S / 2, vi * S / 2, gw_ * ci[:, c], OH // 2, OW // 2) for c in range(3)], -1)
    halo = (cv2.GaussianBlur(gacc, (0, 0), 2.0 * S) + cv2.GaussianBlur(gacc, (0, 0), 6.0 * S) * 0.8) * E
    # rim light: a soft glowing orange-red line along the silhouettes (glow only, no scattered dots)
    RC = L.PAL[int(0.40 * 255)]
    rw = e_[inf] * (0.3 + 0.7 * np.clip(d_[inf] * 1.6, 0, 1)) * 1.6 * RIMK
    racc = L.splat(ui * S / 2, vi * S / 2, rw, OH // 2, OW // 2)
    rl = (cv2.GaussianBlur(racc, (0, 0), 1.2 * S) * 1.0 + cv2.GaussianBlur(racc, (0, 0), 4.0 * S) * 1.2) * E
    halo = halo + rl[..., None] * RC[None, None]
    if GLX is not None:                                           # fabric glint flare
        gacc2 = np.stack([L.splat(GLX[0] * S / 2, GLX[1] * S / 2, GLX[2] * GLX[3][:, c], OH // 2, OW // 2) for c in range(3)], -1)
        halo = halo + (cv2.GaussianBlur(gacc2, (0, 0), 1.3 * S) * 1.1 + cv2.GaussianBlur(gacc2, (0, 0), 4.0 * S) * 0.8) * E
    if TRS is not None and TRS[3] is not None:                     # sparkle glow
        sk = TRS[3]
        sacc = np.stack([L.splat(sk[0] * S / 2, sk[1] * S / 2, sk[3] * sk[2][:, c], OH // 2, OW // 2) for c in range(3)], -1)
        halo = halo + (cv2.GaussianBlur(sacc, (0, 0), 1.4 * S) * 1.2 + cv2.GaussianBlur(sacc, (0, 0), 4.5 * S) * 0.9) * E
    # faint warm floor behind (defocused burgundy, like the benchmark's distance) -- smooth, so design res is enough
    bg = cv2.GaussianBlur((1 - presence) * (0.5 + 0.5 * cv2.GaussianBlur(dep, (0, 0), 40) * 3), (0, 0), 20)
    if TRS is not None:
        bg = bg * TRS[4]                                          # floor glow only once she is formed
    if S != 1:
        bg = cv2.resize(bg, (OW, OH), interpolation=cv2.INTER_CUBIC)
    img = img + bg[..., None] * L.DEEP[None, None] * 0.10
    img = img + cv2.resize(halo, (OW, OH)) * L.EXP * L.GLOW * float(os.environ.get('HALO', '1')) * (0.9 + 0.2 * p)
    vg = L.vig if S == 1 else cv2.resize(L.vig.astype(np.float32), (OW, OH), interpolation=cv2.INTER_CUBIC).reshape(OH, OW, -1)
    img = img * vg
    Lm = luma(img)[..., None]
    sat = 1.0 + 0.30 * np.exp(-((Lm - 0.35) / 0.25) ** 2)         # richer gold in the mids (not olive)
    img = np.clip(Lm + (img - Lm) * sat, 0, None)
    Lm = luma(img)[..., None]
    Lt = Lm / (1 + Lm * 0.85)
    return np.clip(img * (Lt / np.maximum(Lm, 1e-5)) * 1.25, 0, 1)


if __name__ == '__main__':
    out = render(load_depth(sys.argv[1]), int(sys.argv[3]) if len(sys.argv) > 3 else 1)
    cv2.imwrite(sys.argv[2], (out * 255 + 0.5).astype(np.uint8))
