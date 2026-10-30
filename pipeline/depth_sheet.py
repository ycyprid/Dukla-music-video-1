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
_BR = {}
DOTADD = {}          # per-dot additions set by a clip script: 'v' value lift, 'f' twinkle gate
NBASE = {}           # extra (finer) lattice level appended after the first NBASE['n'] dots: randomness of the base dots unchanged
CAM = {}              # animated camera per frame (clip script): dyaw, dtilt (rad), dolly (world), explode
LANDOFF = {}          # landscape: per-dot world offsets for particle flights (dx, dy, dz)
PRESMAP = {}          # optional soft subject alpha for this frame (clip script): anti-aliased edges
SPKOFF = np.zeros(2)  # speck field offset px (set per frame by the clip script: follows the footage's motion)
EXTRA = {}            # extra particles for this frame (set by the clip script): u, v, w, col  (footage coords)


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


def flow_transition(t, u, v, mover, ids, HSH, R):
    """forming for subjects in flight (birds): each dot streams in from off-screen BEHIND its subject, along the
    flight direction, on a gentle S-curve (wake). Tail-end first, each bird at its own moment, own flight time,
    soft overshoot on landing. Trail sparks + landing glints. No dissolve (the subjects fly out of frame)."""
    T_ = TRANS
    r1, r2, r3, r4, r5, r6 = R
    n = u.size
    fd = np.array(T_['fdir'], float); fd /= np.linalg.norm(fd)
    if T_.get('fdir_dot') is not None and T_['fdir_dot'][0].size == n:      # each bird its own flight direction
        fx_, fy_ = T_['fdir_dot']; nn_ = np.hypot(fx_, fy_) + 1e-9
        fd = [fx_ / nn_, fy_ / nn_]
    pd = [-fd[1], fd[0]]
    SURF = T_.get('kind') == 'surface'
    if SURF:
        # a whole moving surface (water): the wave arrives from the upstream edge, so that side lands first
        xk = np.clip(1 - u / W, 0, 1) if fd[0] < 0 else np.clip(u / W, 0, 1)
        streams = 0.5 + 0.5 * np.sin(v / 53.0 + 1.9 * np.sin(u / 141.0)) * np.sin(u / 97.0 + 0.4)
        key = 0.55 * xk + 0.25 * streams + 0.20 * r1
        key_o = 0.55 * xk + 0.25 * (1 - streams) + 0.20 * r5
    else:
        # stagger: per-bird moment + tail-first inside each bird (local grid coords from the dot identity) + random
        Iid = (ids % 1_000_000) // 1000 - 500; Jid = ids % 1000 - 500
        proj = (Iid * fd[0] + Jid * fd[1]) * PITCH_PX
        kd = np.clip((proj + 70) / 140, 0, 1)
        rb = L.hash01(ids // 1_000_000, ids * 0 + 7, 555)               # same for every dot of a bird
        # birds high in the frame are about to fly out of the top: they assemble first (smooth, continuous in v)
        hgt = np.where(np.asarray(fd[1]) < 0, np.clip(v / H, 0, 1), np.clip(1 - v / H, 0, 1)) * T_.get('hgt_k', 1.0)
        key = 0.25 * rb + 0.40 * hgt + 0.20 * kd + 0.15 * r1
    din = T_['in_dur'] * (0.8 + 0.4 * r2)
    tin = T_['in0'] + T_['in_spread'] * key
    fd = [np.asarray(fd[0], float) + np.zeros(n), np.asarray(fd[1], float) + np.zeros(n)]; pd = [-fd[1], fd[0]]
    # distance to leave the frame going backwards along the flight path (+ margin): start fully off-screen
    m = 90.0
    with np.errstate(divide='ignore', invalid='ignore'):
        lx = np.where(-fd[0] > 1e-6, (W + m - u) / -fd[0], np.where(-fd[0] < -1e-6, (u + m) / fd[0], np.inf))
        ly = np.where(-fd[1] > 1e-6, (H + m - v) / -fd[1], np.where(-fd[1] < -1e-6, (v + m) / fd[1], np.inf))
    Ld = np.minimum(lx, ly) + 60 + 160 * r6
    A = (30 + 80 * r3) * np.where(r4 > 0.5, 1.0, -1.0)              # S-curve amplitude (wake)
    SOFT = SURF and T_.get('soft')
    if SOFT:
        Ld = 200 + 250 * r6                                          # materialise in mid-air, a ripple-paced drift
        RW = T_.get('ripple_wl', 320.0); RV = T_.get('ripple_v', 83.0)
        A = (14 + 22 * r3)                                           # undulation amplitude
    if SURF:
        with np.errstate(divide='ignore', invalid='ignore'):          # leaving: on downstream, out of the frame
            ox = np.where(fd[0] > 1e-6, (W + m - u) / fd[0], np.where(fd[0] < -1e-6, (u + m) / -fd[0], np.inf))
            oy = np.where(fd[1] > 1e-6, (H + m - v) / fd[1], np.where(fd[1] < -1e-6, (v + m) / -fd[1], np.inf))
        Lo = np.minimum(ox, oy) + 60 + 160 * r6
        if SOFT:
            Lo = 220 + 260 * r6
        dout = T_['out_dur'] * (0.8 + 0.4 * r5)
        tout = T_['out0'] + T_['out_spread'] * key_o
        def state_out(tt):
            return np.clip((tt - tout) / dout, 0, 1) ** 2.2
    elif T_.get('bird_out'):
        # birds dissolve forwards along their flight: head first, each bird at its own moment, accelerating, fading
        key_o = 0.30 * rb + 0.30 * kd + 0.20 * (1 - hgt) + 0.20 * r5
        key_o = np.clip(1 - key_o, 0, 1) if T_.get('out_headfirst', 1) else key_o
        Lo = T_.get('out_len', 260.0) * (0.6 + 0.8 * r6)
        dout = T_['out_dur'] * (0.8 + 0.4 * r5)
        tout = T_['out0'] + T_['out_spread'] * key_o
        def state_out(tt):
            return np.clip((tt - tout) / dout, 0, 1) ** 1.9
    else:
        state_out = None
    def state(tt):
        xi = (tt - tin) / din
        xc = np.clip(xi, 0, 1)
        prog = 0.85 * (1 - (1 - xc) ** 1.7) + 0.15 * _back_out(xc)       # even speed in flight, soft landing
        return np.where(xi <= 0, 1.0, np.where(xi >= 1, 0.0, 1 - prog))
    def path(tt):
        s_ = state(tt)
        e = _sgnpow(s_, 1.15)
        sc = np.clip(s_, 0, 1)
        if SOFT:
            # ride the ripples: a wave travelling with them (their wavelength, their phase speed), fading at home
            xw = u - fd[0] * Ld * e
            bend = A * sc * np.sin(2 * np.pi * ((xw - fd[0] * RV * tt) / RW + r4)) + 6 * sc * np.sin(2 * np.pi * (0.6 * tt + r5))
        else:
            bend = A * np.sin(np.pi * sc) * (1 - 0.35 * sc) + 10 * np.sin(2 * np.pi * (1.4 * tt + r5)) * 4 * sc * (1 - sc)
        x = u - fd[0] * Ld * e + pd[0] * bend
        y = v - fd[1] * Ld * e + pd[1] * bend
        if state_out is not None:
            so = state_out(tt)
            if SOFT:
                xw = u + fd[0] * Lo * so
                bo = A * so * np.sin(2 * np.pi * ((xw - fd[0] * RV * tt) / RW + r4)) + 6 * so * np.sin(2 * np.pi * (0.6 * tt + r5))
            else:
                bo = -A * np.sin(np.pi * so) * 0.8 + 10 * np.sin(2 * np.pi * (1.2 * tt + r3)) * 4 * so * (1 - so)
            xo = u + fd[0] * Lo * so + pd[0] * bo; yo = v + fd[1] * Lo * so + pd[1] * bo
            out_ = so > 0
            x = np.where(out_, xo, x); y = np.where(out_, yo, y); s_ = np.where(out_, so, s_)
        x = np.where(mover, x, u); y = np.where(mover, y, v); s_ = np.where(mover, s_, 0.0)
        return x, y, s_, np.zeros(n)
    K = 7
    SH = 0.5 if SOFT else 1.0                                                     # ripple pace: normal shutter
    smp = [path(t + (k + 0.5) / K * SH / 25) for k in range(K)]                   # long shutter: streaks along the stream
    x, y, sf, z = smp[K // 2]
    vis = np.clip((1 - np.clip(sf, 0, 1)) / 0.06, 0, 1) * (1 + 0.6 * np.clip(sf, 0, 1))   # a touch brighter in flight
    if T_.get('bird_out') and state_out is not None:              # dissolve: fade as they stream on
        so_ = state_out(t); vo = np.clip((1 - so_) / 0.75, 0, 1)
        vis = vis * vo * vo * (3 - 2 * vo) * (1 + 0.5 * np.sin(np.pi * so_))
    if SOFT:                                                       # materialise / dissolve in mid-air
        si = np.clip(state(t), 0, 1); vin = np.clip((1 - si) / 0.55, 0, 1)
        vo = np.ones(n)
        if state_out is not None:
            so_ = state_out(t); vo = np.clip((1 - so_) / 0.7, 0, 1)
        vis = (vin * vin * (3 - 2 * vin)) * (vo * vo * (3 - 2 * vo)) * (1 + 0.3 * np.clip(sf, 0, 1))
    g_in = np.clip((t - (T_['in0'] + 0.5 * T_['in_spread'])) / (0.9 * T_['in_dur']), 0, 1)
    g_out = 1 - np.clip((t - T_.get('bg_out0', 1e9)) / T_.get('bg_out_dur', 1.0), 0, 1)
    if SURF:
        g_in = np.clip(t / (T_['in_spread'] + T_['in_dur']), 0, 1)
        g_out = 1 - np.clip((t - T_['out0']) / (T_['out_spread'] + 0.6 * T_['out_dur']), 0, 1)
    g = (g_in * g_in * (3 - 2 * g_in)) * (g_out * g_out * (3 - 2 * g_out))
    vis = np.where(mover, vis, g)
    # sparkles: trail sparks on the same streams + landing glints that drift on along the flight
    SU, SV, SW, SR, SCi = [], [], [], [], []
    hosts = mover.copy()
    if hosts.any():
        for gi, lag in enumerate((0.05, 0.11, 0.18, 0.27)):
            pick = hosts & (HSH(300 + gi) < 0.06)
            if not pick.any():
                continue
            ang = 2 * np.pi * HSH(310 + gi); sc_ = 10 + 34 * HSH(320 + gi); ph = HSH(330 + gi); fq = 4 + 5 * HSH(340 + gi)
            px, py, ps, _ = path(t - lag)
            ss = np.clip(state(t - lag), 0, 1)
            al = pick & (ss > 0.04) & (ss < 0.96)
            if al.any():
                env = np.sin(np.pi * ss[al]) ** 0.8 * (0.35 + 0.65 * np.maximum(np.sin(2 * np.pi * (fq[al] * t + ph[al])), 0) ** 6)
                spr = sc_[al] * ss[al]
                SU.append(px[al] + np.cos(ang[al]) * spr); SV.append(py[al] + np.sin(ang[al]) * spr)
                SW.append(env * 1.5); SR.append(np.full(al.sum(), 0.7)); SCi.append(232 + 23 * HSH(350 + gi)[al])
        if state_out is not None:
            for gi, lead in enumerate((0.04, 0.09, 0.15)):
                pick = hosts & (HSH(400 + gi) < 0.05)
                px, py, ps, _ = path(t + lead)
                so_ = state_out(t + lead)
                al = pick & (so_ > 0.04) & (so_ < 0.96)
                if al.any():
                    ang = 2 * np.pi * HSH(410 + gi); ph = HSH(420 + gi); fq = 4 + 5 * HSH(430 + gi)
                    env = np.sin(np.pi * so_[al]) ** 0.8 * (0.35 + 0.65 * np.maximum(np.sin(2 * np.pi * (fq[al] * t + ph[al])), 0) ** 6)
                    spr = (10 + 30 * HSH(440 + gi)[al]) * so_[al]
                    SU.append(px[al] + np.cos(ang[al]) * spr); SV.append(py[al] + np.sin(ang[al]) * spr)
                    SW.append(env * 1.0); SR.append(np.full(al.sum(), 0.7)); SCi.append(232 + 23 * HSH(450 + gi)[al])
        gl = hosts & (HSH(360) < 0.10)
        tau = t - (tin + din)
        al = gl & (tau > 0) & (tau < 0.6)
        if al.any():
            tq = tau[al]
            env = np.exp(-tq / 0.18) * (1 - np.exp(-tq / 0.025))
            SU.append(u[al] + (fd[0] + np.zeros(n))[al] * 80 * tq); SV.append(v[al] + (fd[1] + np.zeros(n))[al] * 80 * tq)
            SW.append(env * 3.0); SR.append(np.full(al.sum(), 0.65)); SCi.append(np.full(al.sum(), 250.0))
    sparks = None
    if SU:
        sparks = (np.concatenate(SU), np.concatenate(SV), L.PAL[np.concatenate(SCi).astype(np.uint8)],
                  np.concatenate(SW), np.concatenate(SR))
    return smp, vis, sf, sparks, g


def hands_transition(t, u, v, mover, R):
    """forming: the subject streams in from off-screen along TRANS['fdir'] (its own direction of entry), the end
    nearest that edge landing first, the tips last; only dots right of in_xmin fly (later arrivals enter with the
    footage). Dissolve: everything peels off outward in all directions from the subject's centre, outer parts first,
    each particle on its own slightly curling path, accelerating and fading in mid-air."""
    T_ = TRANS
    r1, r2, r3, r4, r5, r6 = R
    n = u.size
    fd = np.array(T_['fdir'], float); fd /= np.linalg.norm(fd)
    pd = np.array([-fd[1], fd[0]])
    # ---- forming ----
    xmin = T_.get('in_xmin', 0.0) * W
    gate = np.clip((u - xmin) / 90.0, 0, 1)
    gate = gate * gate * (3 - 2 * gate)
    if T_.get('in_xmin', 0.0) > 0:
        span = max(W - xmin, 1.0)
        xk = np.clip((W - u) / span, 0, 1) if fd[0] < 0 else np.clip(u / span, 0, 1)
    else:                                                        # any direction: the side they enter from lands first
        e_ = u * fd[0] + v * fd[1]
        cs_ = np.array([0, W, 0, W]) * fd[0] + np.array([0, 0, H, H]) * fd[1]
        xk = np.clip(1 - (e_ - cs_.min()) / (np.ptp(cs_) + 1e-6), 0, 1)
    streams = 0.5 + 0.5 * np.sin(v / 61.0 + 1.7 * np.sin(u / 133.0)) * np.sin(u / 89.0 + 0.7)
    key = 0.62 * xk + 0.23 * streams + 0.15 * r1
    din = T_['in_dur'] * (0.8 + 0.4 * r2)
    tin = T_['in0'] + T_['in_spread'] * key
    m = 90.0
    with np.errstate(divide='ignore', invalid='ignore'):
        lx = np.where(-fd[0] > 1e-6, (W + m - u) / -fd[0], np.where(-fd[0] < -1e-6, (u + m) / fd[0], np.inf))
        ly = np.where(-fd[1] > 1e-6, (H + m - v) / -fd[1], np.where(-fd[1] < -1e-6, (v + m) / fd[1], np.inf))
    Ld = np.minimum(lx, ly) + 60 + 220 * r6
    A = (24 + 70 * r3) * np.where(r4 > 0.5, 1.0, -1.0)
    def s_in(tt):
        xi = (tt - tin) / din; xc = np.clip(xi, 0, 1)
        prog = 0.7 * (1 - (1 - xc) ** 1.8) + 0.3 * (1 - (1 - xc) ** 3)            # even flight, soft landing, no overshoot
        return np.where(xi <= 0, 1.0, np.where(xi >= 1, 0.0, 1 - prog)) * (gate > 0.01)
    # ---- dissolve ----
    cx, cy = T_.get('ocx', W / 2), T_.get('ocy', H / 2)
    dx, dy = u - cx, v - cy
    rad = np.hypot(dx, dy) + 1e-3
    radn = np.clip(rad / (np.percentile(rad[mover], 95) if mover.any() else W / 2), 0, 1)
    ang = np.arctan2(dy, dx) + 0.9 * (r3 - 0.5)                                  # all directions, a little scattered
    odx, ody = np.cos(ang), np.sin(ang)
    Lo = (150 + 330 * r6) * (0.7 + 0.5 * radn)                                  # a gentle spread, not an explosion
    curl = (40 + 90 * r4) * np.where(r5 > 0.5, 1.0, -1.0)
    key_o = 0.50 * (1 - radn) + 0.25 * (1 - streams) + 0.25 * r5               # outer parts peel off first
    if T_.get('odir') is not None:                                               # directional exit (e.g. fire: up and away)
        od_ = np.array(T_['odir'], float); od_ /= np.linalg.norm(od_)
        a0 = np.arctan2(od_[1], od_[0]); ang = a0 + T_.get('oscat', 0.6) * (r3 - 0.5)
        odx, ody = np.cos(ang), np.sin(ang)
        ex_ = u * od_[0] + v * od_[1]                                            # distance to leave the frame that way
        with np.errstate(divide='ignore', invalid='ignore'):
            lxo = np.where(odx > 1e-6, (W + 60 - u) / odx, np.where(odx < -1e-6, (u + 60) / -odx, np.inf))
            lyo = np.where(ody > 1e-6, (H + 60 - v) / ody, np.where(ody < -1e-6, (v + 60) / -ody, np.inf))
        Lo = np.minimum(lxo, lyo) * (1.0 + 0.5 * r6) + 40
        lead = np.clip((ex_ - ex_[mover].min()) / (np.ptp(ex_[mover]) + 1e-6), 0, 1) if mover.any() else np.zeros(n)
        key_o = 0.62 * (1 - lead) + 0.23 * (1 - streams) + 0.15 * r5          # the leading side goes first
    dout = T_['out_dur'] * (0.75 + 0.5 * r2)
    tout = T_['out0'] + T_['out_spread'] * key_o
    def s_out(tt):
        return np.clip((tt - tout) / dout, 0, 1) ** 1.7
    def path(tt):
        si = s_in(tt); e = _sgnpow(si, 1.15); sc = np.clip(si, 0, 1)
        bend = A * np.sin(np.pi * sc) * (1 - 0.35 * sc) + 8 * np.sin(2 * np.pi * (1.3 * tt + r5)) * 4 * sc * (1 - sc)
        x = u - fd[0] * Ld * e + pd[0] * bend
        y = v - fd[1] * Ld * e + pd[1] * bend
        so = s_out(tt)
        cb = curl * np.sin(np.pi * 0.5 * so) * so
        xo = u + odx * Lo * so - ody * cb
        yo = v + ody * Lo * so + odx * cb - 25 * so * so                          # the faintest lift as they go
        out_ = so > 0
        x = np.where(out_, xo, x); y = np.where(out_, yo, y); sp = np.where(out_, so, si)
        x = np.where(mover, x, u); y = np.where(mover, y, v); sp = np.where(mover, sp, 0.0)
        return x, y, sp, np.zeros(n)
    K = 7
    smp = [path(t + (k + 0.5) / K * 0.6 / 25) for k in range(K)]
    x, y, sf, z = smp[K // 2]
    si_ = np.clip(s_in(t), 0, 1); so_ = s_out(t)
    vin = np.clip((1 - si_) / 0.06, 0, 1) * (1 + 0.5 * si_)                     # lit for the whole flight (from off-screen)
    vo = np.clip((1 - so_) / 0.65, 0, 1); vo = vo * vo * (3 - 2 * vo)           # fades in mid-air on the way out
    life = 0.35 + 0.65 * r1                                                       # the cloud thins as it spreads:
    vth = np.clip((life - so_) / 0.2, 0, 1)                                      # each particle winks out at its own point
    vis = vin * vo * vth * (1 - 0.45 * so_)
    g_in = np.clip((t - (T_['in0'] + 0.5 * T_['in_spread'])) / (0.9 * T_['in_dur']), 0, 1)
    g_out = 1 - np.clip((t - T_['out0']) / (0.6 * T_['out_spread'] + 0.5 * T_['out_dur']), 0, 1)
    g = (g_in * g_in * (3 - 2 * g_in)) * (g_out * g_out * (3 - 2 * g_out))
    vis = np.where(mover, vis, g)
    # ---- sparkles: trail sparks on the arriving streams, lead sparks on the dispersing ones, landing glints ----
    SU, SV, SW, SR, SCi = [], [], [], [], []
    hosts = np.nonzero(mover)[0]
    if hosts.size:
        hr = np.random.default_rng(4243)
        for gi, lag in enumerate((0.05, 0.11, 0.18, 0.27)):
            pick = hosts[hr.random(hosts.size) < 0.05]
            sa = hr.uniform(0, 2 * np.pi, n); sc_ = hr.uniform(10, 40, n); ph = hr.random(n); fq = hr.uniform(4, 9, n)
            for out, tt in ((False, t - lag), (True, t + lag * 0.6)):
                if not pick.size:
                    continue
                px, py, _, _ = path(tt)
                ss = (s_out(tt) if out else np.clip(s_in(tt), 0, 1))[pick]
                al = (ss > 0.04) & (ss < 0.96)
                if not al.any():
                    continue
                q = pick[al]; sq = ss[al]
                env = np.sin(np.pi * sq) ** 0.8 * (0.35 + 0.65 * np.maximum(np.sin(2 * np.pi * (fq[q] * t + ph[q])), 0) ** 6)
                spr = sc_[q] * sq
                SU.append(px[q] + np.cos(sa[q]) * spr); SV.append(py[q] + np.sin(sa[q]) * spr)
                SW.append(env * (1.0 if out else 1.5)); SR.append(np.full(q.size, 0.7)); SCi.append(232 + hr.random(q.size) * 23)
        gl = hosts[np.random.default_rng(98).random(hosts.size) < 0.10]
        tau = t - (tin[gl] + din[gl])
        al = (tau > 0) & (tau < 0.6) & (gate[gl] > 0.5)
        if al.any():
            q = gl[al]; tq = tau[al]
            env = np.exp(-tq / 0.18) * (1 - np.exp(-tq / 0.025))
            SU.append(u[q] + fd[0] * 70 * tq); SV.append(v[q] + fd[1] * 70 * tq - 20 * tq)
            SW.append(env * 3.0); SR.append(np.full(q.size, 0.65)); SCi.append(np.full(q.size, 250.0))
    sparks = None
    if SU:
        sparks = (np.concatenate(SU), np.concatenate(SV), L.PAL[np.concatenate(SCi).astype(np.uint8)],
                  np.concatenate(SW), np.concatenate(SR))
    return smp, vis, sf, sparks, g


def sweep_vis(x, y, t):
    """breeze specks: blow in as a wave from the top-left (along the wind) and clear the same way at the end"""
    T_ = TRANS
    wd = WIND / (np.linalg.norm(WIND) + 1e-9)
    sx = x * wd[0] + y * wd[1]                                     # coordinate along the wind
    corners = np.array([[0, 0], [W, 0], [0, H], [W, H]], float) @ wd
    a, b = corners.min() - 80, corners.max() + 80
    fin = a + (b - a) * np.clip(t / T_['sw_in_dur'], 0, 1.2)       # front of the arriving field
    fout = a + (b - a) * np.clip((t - T_['sw_out0']) / T_['sw_out_dur'], 0, 1.2)   # back of the leaving field
    vin = np.clip((fin - sx) / 160, 0, 1); vout = np.clip((sx - fout) / 160, 0, 1)
    if t < T_['sw_out0']:
        vout = np.ones_like(sx)
    return (vin * vin * (3 - 2 * vin)) * (vout * vout * (3 - 2 * vout))


def _xdraw(rng, rng2, n, fn, *a):
    nb = NBASE.get('n')
    if not nb or n <= nb:
        return getattr(rng, fn)(*a, n)
    return np.concatenate([getattr(rng, fn)(*a, nb), getattr(rng2, fn)(*a, n - nb)])


def sheet_transition(t, u, v, p_, ids=None):
    nb = NBASE.get('n')
    if nb and u.size > nb and ids is None:     # transition computed on the base dots exactly as before; extras sit still
        smp, vis, sf, sparks, g = sheet_transition(t, u[:nb], v[:nb], p_[:nb])
        ne = u.size - nb
        smp = [(np.concatenate([a_, u[nb:]]), np.concatenate([b_, v[nb:]]), np.concatenate([c_, np.zeros(ne)]), np.concatenate([d_, np.zeros(ne)])) for a_, b_, c_, d_ in smp]
        return smp, np.concatenate([vis, np.full(ne, float(g))]), np.concatenate([sf, np.zeros(ne)]), sparks, g
    """forming / dissolving of the dot sheet as a swirl around her vertical axis.
    Figure dots fly; every path starts / ends a set distance OFF-SCREEN (lower-left in, upper-right out), so the
    first and last frames are black. Faint background dots and the floor glow just fade with the formed fraction.
    Overlap: arrival in streams (low-frequency noise), torso first -> limbs follow through, own flight times, soft
    overshoot on arrival. Sparkles: trail sparks on the same paths + landing / take-off glints."""
    T_ = TRANS
    n = u.size
    if ids is None:
        rr = np.random.default_rng(777)                           # stable per-dot randomness (own stream)
        r1, r2, r3, r4, r5, r6 = (rr.random(n) for _ in range(6))
        HSH = None
    else:                                                         # dots rebuilt every frame: randomness by identity
        HSH = lambda salt: L.hash01(ids, ids // 7 + 3, salt)
        r1, r2, r3, r4, r5, r6 = (HSH(101 + 7 * k_) for k_ in range(6))
    cx, cy = ORBIT.get('cx', W / 2), ORBIT.get('cy', H / 2)
    dirn = T_.get('dir', 1.0)
    mover = p_ > 0.35                                            # the figure flies; the empty field only fades
    if T_.get('mode') == 'flow':
        return flow_transition(t, u, v, mover, ids, HSH, (r1, r2, r3, r4, r5, r6))
    if T_.get('mode') == 'hands':
        return hands_transition(t, u, v, mover, (r1, r2, r3, r4, r5, r6))
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
        s_in = np.where(xi <= 0, 1.0, np.where(xi >= 1, 0.0, 1 - _back_out(xi, T_.get('os', 0.8))))
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
    _dz = push_dz(t)
    if _dz:
        u, v = push_xy(u, v, (Z0 - REL * 0.8) * (D - zc) / D, _dz)
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


def push_dz(t):
    """delicate dolly-in: camera travel so the subject plane magnifies by PUSH over the clip (soft start / soft end)"""
    P = float(os.environ.get('PUSH', '0'))
    if P <= 0:
        return 0.0
    TL = float(os.environ.get('CLIPLEN', '10'))
    x = float(np.clip(t / TL, 0, 1))
    e = 0.6 * x + 0.4 * (0.5 - 0.5 * np.cos(np.pi * x))
    Zs = Z0 - REL * 0.8
    return Zs * (1 - 1 / (1 + P * e))


def push_xy(u, v, Zp, dz):
    """screen-space dolly about the screen centre for a point at camera distance Zp (nearer -> moves more)"""
    if dz == 0:
        return u, v
    m = Zp / np.maximum(Zp - dz, 1e-3)
    return W / 2 + (u - W / 2) * m, H / 2 + (v - H / 2) * m


def _proj(gx, gy, d):
    X = (gx - W / 2) / H; Y = -(gy - H / 2) / H; Z = Z0 - REL * d
    ct, st, cy_, sy_ = np.cos(TILT), np.sin(TILT), np.cos(YAW), np.sin(YAW)
    Zc = Z - Z0 * 0.75; X1 = X * cy_ + Zc * sy_; Z1 = -X * sy_ + Zc * cy_
    Y2 = Y * ct - Z1 * st; Z2 = Y * st + Z1 * ct + Z0 * 0.75
    return W / 2 + FOC * X1 / Z2, H / 2 - FOC * Y2 / Z2


def fit_frame(u, v):
    """uniform zoom + shift so the footage's frame edges (at the subject's depth) land just outside the screen"""
    if not _FIT and os.environ.get('FIT') == 'cover':
        # smallest zoom (>= 1) + shift so the footage frame covers the screen at the subject depth (conservative corners)
        d = float(os.environ.get('FIT_D', '0.5')); over = float(os.environ.get('FIT_OVER', '1.015'))
        pu, pv = _proj(np.array([0, W, 0, W], float), np.array([0, 0, H, H], float), d)
        li, ri = max(pu[0], pu[2]), min(pu[1], pu[3]); ti, bi = max(pv[0], pv[1]), min(pv[2], pv[3])
        s_ = max(W / (ri - li), H / (bi - ti)) * over
        if s_ <= 1:
            s_ = 1.0
            cu = float(np.clip(W / 2, li + W / 2, ri - W / 2)); cv_ = float(np.clip(H / 2, ti + H / 2, bi - H / 2))
        else:
            cu, cv_ = (li + ri) / 2, (ti + bi) / 2
        _FIT.update(cu=cu, cv=cv_, s=s_)
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
    gx_, gy_ = gxs.ravel().astype(np.float32), gys.ravel().astype(np.float32)
    if levels == 'deep':                         # + level 3: centres of the half-pitch cells (appended: base order kept)
        gx3, gy3 = np.meshgrid(xs[:-1] + PITCH_PX / 4, ys[:-1] + PITCH_PX / 4)
        gx_ = np.concatenate([gx_, gx3.ravel().astype(np.float32)]); gy_ = np.concatenate([gy_, gy3.ravel().astype(np.float32)])
        lv = np.concatenate([lv, np.full(gx3.size, 3)])
    return gx_, gy_, lv, gxs.shape


def level_vis(lv, scale):
    ss = lambda a, b, x: np.clip((x - a) / (b - a), 0, 1) ** 2 * (3 - 2 * np.clip((x - a) / (b - a), 0, 1))
    return np.choose(lv, [np.ones_like(scale), ss(1.22, 1.45, scale), ss(1.72, 2.0, scale), ss(2.35, 2.8, scale)]).astype(np.float32)


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


def render(dep, seed=1, t=0.0, pts=None, vis=None, ids=None, state=None, lum=None, fab=None):
    FL = float(os.environ.get('FLOOR', '0'))     # cut faint depth (smear trails) before lifting the mids
    if FL > 0:
        dep = np.clip((dep - FL) / (1 - FL), 0, 1)
    if GAMMA != 1:
        dep = dep ** GAMMA
    p = soft_pulse(t) * float(os.environ.get('PULSEK', '1'))      # 0: no beat pulse (full-frame sheets)
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
    rs = RIMSCALE.setdefault('s', float(os.environ['RIMS']) if os.environ.get('RIMS') else np.percentile(eg, 97) + 1e-6)          # fixed per clip (no flicker)
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
    if lum is not None and PRESMAP.get('a') is not None:
        # edge band: tone from just inside the subject (the footage's darker silhouette skin would draw a red outline)
        ak = int(PITCH_PX * 2) | 1
        vin_ = cv2.dilate(val, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ak, ak)))
        a_ = PRESMAP['a']
        a3 = a_ ** 3
        val = val * a3 + vin_ * (1 - a3)
    presence = np.clip((dep - 0.06) / 0.30, 0, 1) ** 1.3          # ground fades out to black
    if lum is not None:
        presence = np.clip((dep - 0.02) / 0.10, 0, 1)               # picture mode: inside the figure = fully present
        if PRESMAP.get('a') is not None:                            # soft edge supplied (anti-aliased outline on the dot grid)
            PRESMAP['p0'] = presence                                 # (colour stays the subject's own at the edge)
            presence = np.minimum(presence, PRESMAP['a'])

    # lattice in world plane (with margin so the tilt never exposes an edge)
    if pts is None:
        pts = lattice()                                   # dots ride on the footage when pts are advected
    gxs, gys = pts
    samp = sampler(pts)
    d_, v_, p_, f_, e_ = samp(dep), samp(val), samp(presence), samp(fac), samp(rim)
    if DOTADD.get('v') is not None and DOTADD['v'].size == d_.size:     # per-dot value lift (e.g. a backlit rim)
        v_ = np.clip(v_ + DOTADD['v'], 0, 1)
    FAB = os.environ.get('FABRIC') and (lum is not None or fab is not None)
    if FAB:
        if fab is not None:
            fmap = fab                                                            # detail map supplied (e.g. feathers)
        else:
            # where the footage has fine, bright, high-contrast detail (lace, satin highlights, tulle edges)
            lm_ = np.clip(lum, 0, 1.2).astype(np.float32)
            hp = np.abs(lm_ - cv2.GaussianBlur(lm_, (0, 0), 2.2))                # fine local contrast
            fmap = np.clip(hp / float(os.environ.get('FAB_HP', '0.07')), 0, 1) ** 1.2 * np.clip((lm_ - 0.30) / 0.40, 0, 1)
        fd_ = samp(fmap)
        if DOTADD.get('f') is not None and DOTADD['f'].size == fd_.size:
            fd_ = np.maximum(fd_, DOTADD['f'])
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
    if CAM.get('explode', 0) > 0:              # relief splits into layers that drift apart towards the camera
        NL = CAM.get('layers', 7)
        li = np.floor(np.clip(d_, 0, 0.999) * NL)
        Z = Z - CAM['explode'] * CAM.get('gap', 0.09) * li * (d_ > 0.05)
    BFADE = None
    if CAM.get('burst', 0) > 0:                # fly-through: the bouquet bursts outward around the camera's path
        b_ = CAM['burst']; nb_ = X.size
        if _BR.get('n') != nb_:
            rb_ = np.random.default_rng(4242); _BR.update(n=nb_, r1=rb_.random(nb_), r2=rb_.random(nb_), r3=rb_.random(nb_), r4=rb_.random(nb_))
        ax_, ay_ = (CAM['aim'][0], CAM['aim'][1]) if CAM.get('aim') else (0.0, 0.0)
        X = X - ax_; Y = Y - ay_                 # disperse around the camera's path (the rose it flies into)
        rad_ = np.hypot(X, Y) + 0.03
        if CAM.get('stag'):                    # staggered release: outer petals peel first, the heart last; each eases out of rest
            s0_, sp_, du_ = CAM['stag']
            if 'key' not in _BR:
                pm_ = p_ > 0.5
                cxw, cyw = (float(np.average(X[pm_], weights=v_[pm_] + 0.05)), float(np.average(Y[pm_], weights=v_[pm_] + 0.05))) if pm_.any() else (0.0, 0.0)
                rr_ = np.hypot(X - cxw, Y - cyw); r95 = np.percentile(rr_[pm_], 95) if pm_.any() else 1.0
                _BR['key'] = 0.72 * (1 - np.clip(rr_ / r95, 0, 1)) + 0.28 * _BR['r4']      # 0 = outer edge .. 1 = heart
            x_ = np.clip((CAM['bt'] - (s0_ + sp_ * _BR['key'])) / (du_ * (0.8 + 0.4 * _BR['r2'])), 0, 1.6)
            b_ = x_ ** 1.25
            fd_ = np.clip((x_ - (0.65 + 0.35 * _BR['r3'])) / 0.45, 0, 1)
            BFADE = 1 - fd_ * fd_ * (3 - 2 * fd_)                              # each particle thins out at its own moment
        spr_ = b_ ** 1.6 * (0.25 + 1.5 * _BR['r1']) * float(os.environ.get('BURST_S', '1'))
        X = X + X / rad_ * spr_; Y = Y + Y / rad_ * spr_ * 0.9 + 0.15 * float(os.environ.get('BURST_S', '1')) * b_ ** 2 * (_BR['r3'] - 0.4)
        Z = Z - b_ ** 1.5 * (0.08 + float(os.environ.get('BURST_Z', '0.7')) * _BR['r2'])
        X = X + ax_; Y = Y + ay_
    LAND = os.environ.get('LAND')
    if LAND:                                   # landscape: the sheet image stretched over a large ground plane
        SXL, SYL = (float(x_) for x_ in LAND.split(','))
        X = X * SXL; Y = Y * SYL + float(os.environ.get('LAND_Y0', '0'))
        if LANDOFF.get('dy') is not None:            # particles in flight (forming / dispersing), world units
            Y = Y + LANDOFF['dy']; Z = Z - LANDOFF['dz']; X = X + LANDOFF.get('dx', 0)
    # camera rotation (tilt about x, yaw about y)
    TL_, YW_ = TILT + CAM.get('dtilt', 0.0), YAW + CAM.get('dyaw', 0.0)       # animated camera (orbit / rise)
    ct, st, cy_, sy_ = np.cos(TL_), np.sin(TL_), np.cos(YW_), np.sin(YW_)
    Zc = Z - Z0 * 0.75                                               # rotate around a point in the scene
    X1 = X * cy_ + Zc * sy_; Z1 = -X * sy_ + Zc * cy_
    Y2 = Y * ct - Z1 * st; Z2 = Y * st + Z1 * ct + Z0 * 0.75
    if CAM.get('aim'):                         # camera trucks / pedestals so its path heads into a rose's heart
        axt, ayt, azt, ak_ = CAM['aim']
        zct = azt - Z0 * 0.75
        x1t = axt * cy_ + zct * sy_; z1t = -axt * sy_ + zct * cy_
        y2t = ayt * ct - z1t * st
        X1 = X1 - ak_ * x1t; Y2 = Y2 - ak_ * y2t
    if CAM.get('dolly', 0) > 0:                # a real push-in: the camera travels towards the subject
        CAMF = np.clip((Z2 - CAM['dolly'] - 0.06) / 0.25, 0, 1)      # past the lens: gone
        Z2 = np.maximum(Z2 - CAM['dolly'], 0.05)
    u = W / 2 + FOC * X1 / Z2 / (H / H); v = H / 2 - FOC * Y2 / Z2
    PERS = None
    if LAND:                                   # far dots smaller, and fading into the dark at the horizon
        PERS = np.clip(float(os.environ.get('LAND_ZREF', '2.0')) / np.maximum(Z2, 0.05), 0.05, 1.6)
        v = v + float(os.environ.get('LAND_V', '0'))
    PERS_E = None
    u = W / 2 + (u - W / 2) * 1.0; v = v
    if os.environ.get('FIT'):                     # fit the footage frame to the screen (+ overscan): no margin shows
        u, v = fit_frame(u, v)
    DZ = push_dz(t)
    if DZ:
        u, v = push_xy(u, v, Z2, DZ)
    TRS = None
    if TRANS:
        TRS = sheet_transition(t, u, v, p_, ids)

    col = L.PAL[((0.06 + 0.94 * v_) * 255).astype(np.uint8)]
    fog = np.clip(1 - (samp(PRESMAP['p0']) if (lum is not None and PRESMAP.get('a') is not None) else p_) * 1.4, 0, 1)[:, None]
    col = col * (1 - fog * 0.6) + L.DEEP * fog * 0.6
    if os.environ.get('HOTW'):                    # white-hot: the brightest parts turn white-silver (fire core)
        h0_, hk_ = (float(x_) for x_ in os.environ['HOTW'].split(','))
        hw_ = np.clip((v_ - h0_) / (1 - h0_ + 1e-6), 0, 1); hw_ = (hw_ * hw_ * (3 - 2 * hw_) * hk_)[:, None]
        col = col * (1 - hw_) + np.array([0.97, 0.97, 0.95], np.float32)[None] * hw_
    pfl = p_ if (lum is not None and PRESMAP.get('a') is not None) else (0.10 + 0.90 * p_)   # soft-edge mode: nothing outside
    w = (0.25 + 0.75 * v_) * pfl * 3.4 * (0.35 + 0.65 * f_ ** 2) * float(os.environ.get("GAIN", "1"))
    if CAM.get('dolly', 0) > 0:
        w = w * CAMF
    if BFADE is not None:
        w = w * BFADE
    if os.environ.get('DBGBURST') and BFADE is not None:
        on_ = (u > 0) & (u < W) & (v > 0) & (v < H)
        print('t %.2f  dots %d  onscreen %.2f  bfade>0.5 %.2f  camf>0.5 %.2f  both+on %.3f  Z2 p10/50 %.2f %.2f' % (t, u.size, on_.mean(), (BFADE > 0.5).mean(), (CAMF > 0.5).mean(), (on_ & (BFADE > 0.5) & (CAMF > 0.5) & (p_ > 0.3)).mean(), np.percentile(Z2, 10), np.percentile(Z2, 50)))
    if LAND:
        w = w * np.exp(-np.maximum(Z2 - float(os.environ.get('LAND_FOG0', '3')), 0) / float(os.environ.get('LAND_FOG', '4')))
        w = w * np.clip(PERS, 0, 1) ** float(os.environ.get('LAND_DENS', '1.0'))      # far dots crowd together: dim them
        w = w * np.clip((Z2 - 0.15) / 0.25, 0, 1)                                      # nothing behind / at the lens
    if os.environ.get('DBGDOTS'):
        np.savez(os.environ['DBGDOTS'], u=u, v=v, w=w, val=v_, p=p_, d=d_, e=e_, col=col)
    if os.environ.get('DARKCUT'):                 # dark areas fall to black instead of a dim red screen
        lo_, hi_ = (float(x_) for x_ in os.environ['DARKCUT'].split(','))
        dc_ = np.clip((v_ - lo_) / (hi_ - lo_), 0, 1); w = w * dc_ * dc_ * (3 - 2 * dc_)
    # twinkle glints: sparse ground-anchored dots, strongest on bright parts
    if ids is None:
        rgx = np.random.default_rng(seed + 999)
        sel = (_xdraw(rng, rgx, u.size, 'random') < 0.03) * _xdraw(rng, rgx, u.size, 'uniform', 0.5, 1.0)
        ph = _xdraw(rng, rgx, u.size, 'random'); kk = _xdraw(rng, rgx, u.size, 'uniform', 0.35, 0.9)        # each glint: own phase / rate (Hz)
    else:                                                                    # glints tied to dot identity
        hh = lambda salt: L.hash01(ids, ids // 7 + 3, salt)
        sel = (hh(11) < 0.03) * (0.5 + 0.5 * hh(23)); ph = hh(37); kk = 0.35 + 0.55 * hh(41)
    LQ = float(os.environ.get('LOOPQ', '0'))          # seamless loops: every rhythm a whole number of cycles per loop
    if LQ > 0:
        kk = np.maximum(np.round(kk * LQ), 1) / LQ
    tw = sel * np.maximum(np.sin(2 * np.pi * (kk * t + ph)), 0) ** 24 * (0.3 + 0.7 * v_) * p_
    if FAB:
        # fabric twinkles: a dot can only flash where real detail sits under it; own threshold + rhythm per dot
        if ids is None:
            fr_ = np.random.default_rng(31337)
            fx_ = np.random.default_rng(31338)
            thr = _xdraw(fr_, fx_, u.size, 'uniform', 0.15, 0.85); ph2 = _xdraw(fr_, fx_, u.size, 'random'); kk2 = _xdraw(fr_, fx_, u.size, 'uniform', 0.9, 2.4)
        else:                                                                 # identity-stable (rebuilt dots)
            hh2 = lambda salt: L.hash01(ids, ids // 5 + 11, salt)
            thr = 0.15 + 0.70 * hh2(71); ph2 = hh2(73); kk2 = 0.9 + 1.5 * hh2(79)
        if LQ > 0:
            kk2 = np.maximum(np.round(kk2 * LQ), 1) / LQ
        gate = np.clip((fd_ - thr) / 0.18, 0, 1); gate = gate * gate * (3 - 2 * gate)
        tw = np.maximum(tw * 0.4, gate * np.maximum(np.sin(2 * np.pi * (kk2 * t + ph2)), 0) ** 20 * p_)
    if vis is not None:
        tw = tw * np.clip(vis, 0, 1)                   # no glints on dots that aren't there yet
    w = w * (1 + float(os.environ.get('TWK', '2.2')) * tw)
    if vis is not None:
        w = w * vis
    col = col * (1 - tw[:, None] * 0.5) + L.PAL[255][None] * tw[:, None] * 0.5
    # depth of field: focus on the front of the flower
    zf = Z0 - REL * 0.82
    if CAM.get('focus') is not None:            # focus pull: the focus plane follows the subject as the camera travels
        zf = max(zf - CAM['focus'], 0.35)
    r = np.maximum(0.55, 26.0 * DOFK * np.abs(1 / Z2 - 1 / zf) * Z0)
    if TRS is not None:
        smp, tvis, tsf, SPK_, GFORM = TRS
        tsf = np.clip(tsf, 0, 1)
        w = w * tvis * (1 + 0.35 * tsf)                           # in flight: a touch brighter (liquid sparks)
        r = r + float(TRANS.get('fly_soft', 3.5)) * tsf ** 1.5      # and softer
        col = col * (1 - 0.35 * tsf[:, None]) + L.PAL[int(0.82 * 255)][None] * 0.35 * tsf[:, None]
        u, v = smp[len(smp) // 2][0], smp[len(smp) // 2][1]
    inf = (u > -60) & (u < W + 60) & (v > -60) & (v < H + 60)
    if os.environ.get('SCREENOCC'):
        # presence where the dots are DRAWN (after 3D projection / fit / flight), for the floor glow and the
        # specks hidden behind the subject -> no ghost hole at the footage position
        pw = p_ * (TRS[1] if TRS is not None else 1.0)
        ps_ = L.splat(np.clip(u, -1, W), np.clip(v, -1, H), np.where(inf, pw, 0.0), H, W)
        presence = np.clip(cv2.GaussianBlur(ps_, (0, 0), PITCH_PX * 0.8) * PITCH_PX * PITCH_PX * 1.15, 0, 1).astype(np.float32)
    if TRS is not None:                                           # motion blur along each dot's flight
        live = inf & (w > 1e-4)
        K_ = len(smp)
        U = np.concatenate([sm_[0][live] for sm_ in smp]); V = np.concatenate([sm_[1][live] for sm_ in smp])
        C = np.tile(col[live], (K_, 1)); Wt = np.tile(w[live] / K_, K_); Rr = np.tile(r[live], K_)
        NSH = U.size; VAL_E = np.tile(v_[live], K_); RAW_R = Rr.copy(); PRES_E = np.tile(p_[live], K_)
        if SPK_ is not None:                                      # sparkle cores
            U = np.concatenate([U, SPK_[0]]); V = np.concatenate([V, SPK_[1]]); C = np.concatenate([C, SPK_[2]])
            Wt = np.concatenate([Wt, SPK_[3]]); Rr = np.concatenate([Rr, SPK_[4]])
    else:
        U, V, C, Wt, Rr = u[inf], v[inf], col[inf], w[inf], r[inf]
        PERS_E = PERS[inf] if PERS is not None else None
        NSH = U.size; VAL_E = v_[inf]; PRES_E = p_[inf]
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
            dkp = 0.55 + 0.9 * dr / 9                                    # parallax: nearer (bigger) specks follow more
            du = np.mod(du0 + vx * t + SPKOFF[0] * dkp, W); dv = np.mod(dv0 + vy * t + SPKOFF[1] * dkp, H)
        _dz = push_dz(t) + CAM.get('dolly', 0.0)          # specks rush past when the camera travels
        if _dz:
            du, dv = push_xy(du, dv, Z0 * (0.62 - 0.12 * dr / 9), _dz)     # near dust: rushes past a little faster
        dcol = np.repeat(L.PAL[int(0.80 * 255)][None], ND, 0)
        dw = rng.uniform(0.15, 0.55, ND) * 0.55 * (0.6 + 0.4 * np.sin(rng.uniform(0, 6.28, ND) + 1.3 * t) ** 2)
        SPM = 1.0
        if TRS is not None and TRANS.get('kind') == 'surface':
            SPM = float(TRS[4])
        if os.environ.get('SPECKFADE'):
            fi_, fo_ = (float(x_) for x_ in os.environ['SPECKFADE'].split(','))
            TL_ = float(os.environ.get('CLIPLEN', '1e9'))
            a_ = np.clip(t / fi_, 0, 1); b_ = np.clip((TL_ - 0.12 - t) / fo_, 0, 1)
            SPM *= (a_ * a_ * (3 - 2 * a_)) * (b_ * b_ * (3 - 2 * b_))
        if os.environ.get('NOSPECKS'):
            SPM = 0.0
        dw = dw * SPM
        if wind and TRANS.get('sw_in_dur'):
            dw = dw * sweep_vis(du, dv, t)
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
            bkp = 0.25 + 0.75 * bz
            bu = np.mod(bu0 + bjx * bz * t + SPKOFF[0] * bkp, W); bv = np.mod(bv0 - bjy * bz * t + SPKOFF[1] * bkp, H)
        if _dz:
            bu, bv = push_xy(bu, bv, Z0 * (1.0 + 0.8 * (1 - bz)), _dz)    # far layer: barely moves
        bph = rng.uniform(0, 6.28, NB); bk = rng.uniform(0.3, 0.8, NB)
        btw = 0.35 + 0.65 * np.maximum(np.sin(bph + 2 * np.pi * bk * t), 0) ** 3
        occl = cv2.GaussianBlur(presence, (0, 0), 6)
        bvis = 1 - np.clip(cv2.remap(occl, bu.astype(np.float32).reshape(1, -1), bv.astype(np.float32).reshape(1, -1), cv2.INTER_LINEAR)[0] * 1.6, 0, 1)
        bci = (0.55 + 0.35 * rng.random(NB)) * 255
        bcol = L.PAL[bci.astype(np.uint8)]
        bw = rng.uniform(0.25, 1.0, NB) ** 2 * 2.3 * btw * bvis * SPM
        br = 0.6 + 1.6 * (1 - bz) * rng.random(NB) + (rng.random(NB) < 0.06) * rng.uniform(3, 8, NB)   # a few soft bokeh                     # mostly crisp, a few soft
        if wind and TRANS.get('sw_in_dur'):
            bw = bw * sweep_vis(bu, bv, t)
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
        if PRESMAP.get('a') is not None:                              # edge dots: area follows coverage (anti-aliased outline)
            pe_ = np.ones(U.size); pe_[:NSH] = PRES_E
            Rd = Rd * np.sqrt(np.clip(pe_, 0.04, 1))
        if os.environ.get('LAND') and PERS_E is not None:
            ps_ = np.ones(U.size); ps_[:NSH] = PERS_E
            Rd = Rd * ps_
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
    if vis is not None and os.environ.get('RIMVIS'):
        rw = rw * np.clip(vis[inf], 0, None)
    if NBASE.get('n') and u.size > NBASE['n'] and vis is not None:     # finer level: rim only as it fades in
        xv_ = np.ones(u.size); xv_[NBASE['n']:] = np.clip(vis[NBASE['n']:], 0, 1); rw = rw * xv_[inf]
    if TRS is not None and TRANS.get('soft'):                     # rim only where the dots are already visible
        rw = rw * np.clip(TRS[1][inf], 0, 1)
    racc = L.splat(ui * S / 2, vi * S / 2, rw, OH // 2, OW // 2)
    rl = (cv2.GaussianBlur(racc, (0, 0), 1.2 * S) * 1.0 + cv2.GaussianBlur(racc, (0, 0), 4.0 * S) * 1.2) * E
    halo = halo + rl[..., None] * RC[None, None]
    if EXTRA.get('u') is not None and len(EXTRA['u']):            # e.g. the leaf's trailing sparkles
        eu, ev = EXTRA['u'].astype(np.float64), EXTRA['v'].astype(np.float64)
        if os.environ.get('FIT'):
            eu, ev = fit_frame(eu, ev)
        _dze = push_dz(t)
        if _dze:
            eu, ev = push_xy(eu, ev, EXTRA.get('z', Z0 - REL * 0.8), _dze)
        eacc = np.stack([L.splat(eu * S / 2, ev * S / 2, EXTRA['w'] * EXTRA['col'][:, c], OH // 2, OW // 2) for c in range(3)], -1)
        halo = halo + (cv2.GaussianBlur(eacc, (0, 0), 1.3 * S) * 1.0 + cv2.GaussianBlur(eacc, (0, 0), 4.0 * S) * 0.7) * E
        ecore = np.stack([L.splat(eu * S, ev * S, EXTRA['w'] * EXTRA['col'][:, c], OH, OW) for c in range(3)], -1)
        img = img + cv2.GaussianBlur(ecore, (0, 0), 0.75 * S) * E * L.EXP * 1.1
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
    if os.environ.get('SPECKFADE'):                               # shots without transitions: glow fades at the ends too
        fi_, fo_ = (float(x_) for x_ in os.environ['SPECKFADE'].split(','))
        TL_ = float(os.environ.get('CLIPLEN', '1e9'))
        a_ = np.clip(t / fi_, 0, 1); b_ = np.clip((TL_ - 0.12 - t) / fo_, 0, 1)
        bg = bg * (a_ * a_ * (3 - 2 * a_)) * (b_ * b_ * (3 - 2 * b_))
    if S != 1:
        bg = cv2.resize(bg, (OW, OH), interpolation=cv2.INTER_CUBIC)
    img = img + bg[..., None] * L.DEEP[None, None] * 0.10
    img = img + cv2.resize(halo, (OW, OH)) * L.EXP * L.GLOW * float(os.environ.get('HALO', '1')) * (0.9 + 0.2 * p)
    vg = L.vig if S == 1 else cv2.resize(L.vig.astype(np.float32), (OW, OH), interpolation=cv2.INTER_CUBIC).reshape(OH, OW, -1)
    img = img * vg
    if os.environ.get('SHOTFADE'):                                # whole-image fade at the shot ends (black first / last frame)
        fi2, fo2 = (float(x_) for x_ in os.environ['SHOTFADE'].split(','))
        TL2 = float(os.environ.get('CLIPLEN', '1e9'))
        ka = np.clip(t / fi2, 0, 1) if fi2 > 0 else 1.0
        kb = np.clip((TL2 - 0.04 - t) / fo2, 0, 1) if fo2 > 0 else 1.0
        img = img * (ka * ka * (3 - 2 * ka)) * (kb * kb * (3 - 2 * kb))
    Lm = luma(img)[..., None]
    sat = 1.0 + 0.30 * np.exp(-((Lm - 0.35) / 0.25) ** 2)         # richer gold in the mids (not olive)
    img = np.clip(Lm + (img - Lm) * sat, 0, None)
    Lm = luma(img)[..., None]
    Lt = Lm / (1 + Lm * 0.85)
    return np.clip(img * (Lt / np.maximum(Lm, 1e-5)) * 1.25, 0, 1)


if __name__ == '__main__':
    out = render(load_depth(sys.argv[1]), int(sys.argv[3]) if len(sys.argv) > 3 else 1)
    cv2.imwrite(sys.argv[2], (out * 255 + 0.5).astype(np.uint8))
