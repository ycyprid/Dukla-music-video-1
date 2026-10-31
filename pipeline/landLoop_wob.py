"""Landscape loop in the footage dot look. 15 s, 25 fps.
LOOPMODE=seamless : every motion repeats exactly at T (glide crossfaded over one loop length, rhythms quantised)
LOOPMODE=trans    : same motion; particles stream in from behind the camera along the ground at the start,
                    and stream away towards the horizon at the end (near ground first both times)
usage: python3 landLoop.py outdir f0 f1"""
import sys, os, numpy as np, cv2
sys.path.insert(0, '/home/claude/mv'); os.chdir('/home/claude/mv')
import depth_sheet as DS
from common import W, H
outdir = sys.argv[1]; f0, f1 = int(sys.argv[2]), int(sys.argv[3]); os.makedirs(outdir, exist_ok=True)
T = float(os.environ.get('LOOPT', '15')); FPS = 25
MODE = os.environ.get('LOOPMODE', 'seamless')
SX, SY = (float(x) for x in os.environ['LAND'].split(','))
Y0 = float(os.environ.get('LAND_Y0', '0'))
P = DS.PITCH_PX
K_ROWS = int(round(float(os.environ.get('GLIDE', '10')) * T / P))
D = K_ROWS * P                                     # glide distance per loop: a whole number of dot rows
GL = D / T
SC = float(os.environ.get('SW_SCALE', '9')); DA = float(os.environ.get('DET_A', '0.035'))
REL_ = float(os.environ.get('REL', '0.3'))
yy0, xx0 = np.mgrid[0:H, 0:W].astype(np.float32)
FIELDS = {}


def tex(scale, seed):
    if (scale, seed) not in FIELDS:
        r_ = np.random.default_rng(seed); gw, gh = 256, 512
        z = r_.normal(size=(gh, gw)).astype(np.float32)
        z = cv2.GaussianBlur(z, (0, 0), scale); z /= z.std() + 1e-6
        FIELDS[(scale, seed)] = z
    return FIELDS[(scale, seed)]


def field(scale, seed, xw, yw):
    z = tex(scale, seed); gh, gw = z.shape
    fx = ((xw / (SX * W / H) + 0.5) * gw).astype(np.float32); fy = ((yw / SY + 0.5) * gh).astype(np.float32)
    return cv2.remap(z, fx, fy, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT101)


# ---- wobble ramp (non-looping versions): from WOB_T0 (absolute s) amplitude x WOB_A and wave speed x WOB_S,
#      eased in over one bar; everything else keeps the loop's own time
WOB_T0 = float(os.environ.get('WOB_T0', '1e9')); WOB_A = float(os.environ.get('WOB_A', '2.0')); WOB_S = float(os.environ.get('WOB_S', '1.5'))
WOB_BAR = float(os.environ.get('WOB_BAR', str(4 * 60 / 146)))
_tg = np.linspace(0, 60, 60001)
_rg = np.clip((_tg - WOB_T0) / WOB_BAR, 0, 1); _rg = _rg * _rg * (3 - 2 * _rg)
_tau = _tg + (WOB_S - 1) * np.cumsum(_rg) * (_tg[1] - _tg[0])
def wob(t):
    return float(np.interp(t, _tg, _rg)), float(np.interp(t, _tg, _tau))


def surface(t, off, tm=None):
    """height + detail for ground offset `off` (sheet px) at time t (morph phases periodic in T)"""
    xw = (xx0 - W / 2) / H * SX; yw = -((yy0 - off) - H / 2) / H * SY
    tm = t if tm is None else tm                       # morph time (sped up by the wobble)
    ph = 2 * np.pi * tm / T
    h = field(SC, 3, xw, yw) * np.cos(ph) + field(SC, 13, xw, yw) * np.sin(ph) \
        + 0.25 * (field(SC * 0.45, 4, xw, yw) * np.cos(2 * ph) + field(SC * 0.45, 14, xw, yw) * np.sin(2 * ph))
    warp = field(2.5, 9, xw, yw) * 0.35
    silk = np.sin(2 * np.pi * (xw * 0.35 + yw * 1.0) / float(os.environ.get('DET_WL', '0.55')) + warp * 4 - 2 * np.pi * tm / T)
    grain = field(1.2, 11, xw, yw)
    return h, 0.7 * silk + 0.3 * grain


def nv(*a):
    a = np.array(a, float); return a / np.linalg.norm(a)


Lk = nv(-0.6, -0.35, 0.72); Vc = nv(0, -0.75, 0.66)
gx0, gy0 = DS.lattice()
col_id = np.round((gx0 - gx0.min()) / P).astype(np.int64); row0 = np.round((gy0 - gy0.min()) / P).astype(np.int64)
# ---- transition setup: per-dot screen depth key from a probe render (near ground = bottom of screen = first) ----
if MODE == 'trans':
    IN0, ISP, IDU = 0.0, float(os.environ.get('IN_SPREAD', '1.6')), float(os.environ.get('IN_DUR', '1.5'))
    ODU, OSP = float(os.environ.get('OUT_DUR', '1.6')), float(os.environ.get('OUT_SPREAD', '1.6'))
    OUT0 = T - (OSP + ODU) - 0.2


def flights(t, gidx_row, key):
    """per-dot world offsets (dx, dy, dz) and visibility at time t. Randomness tied to the dot's ground row."""
    hsh = lambda salt: DS.L.hash01(col_id, gidx_row, salt)
    r1, r2, r3, r4, r5, r6 = (hsh(500 + 7 * k) for k in range(6))
    dx = np.zeros(key.size); dy = np.zeros(key.size); dz = np.zeros(key.size); vis = np.ones(key.size)
    # forming: from behind the camera, forward along the ground, landing near-first
    tin = IN0 + ISP * (0.8 * key + 0.2 * r1); din = IDU * (0.8 + 0.4 * r2)
    s = np.clip((t - tin) / din, 0, 1); e = 1 - (1 - s) ** 2.2
    Dist = 1.6 + 2.4 * r3
    fly_in = s < 1
    dy = np.where(fly_in, -Dist * (1 - e), dy)
    dz = np.where(fly_in, (0.10 + 0.25 * r4) * np.sin(np.pi * e) * (1 - e * 0.3), dz)
    dx = np.where(fly_in, (r5 - 0.5) * 0.8 * (1 - e), dx)
    vis = np.where(t < tin, 0.0, np.clip(s / 0.06, 0, 1)) * (1 + 0.5 * (fly_in & (s > 0)))
    # dispersing: lift off near-first, stream away towards the horizon, fading
    tout = OUT0 + OSP * (0.8 * key + 0.2 * r6); dout = ODU * (0.8 + 0.4 * r2)
    so = np.clip((t - tout) / dout, 0, 1); eo = so ** 1.8
    out_ = so > 0
    Do = 3.0 + 4.0 * r3
    dy = np.where(out_, Do * eo, dy)
    dz = np.where(out_, (0.08 + 0.3 * r4) * eo, dz)
    dx = np.where(out_, (r5 - 0.5) * 0.6 * eo, dx)
    vo = np.clip((0.85 - so) / 0.6, 0, 1); vo = vo * vo * (3 - 2 * vo)
    vis = np.where(out_, vis * vo * (1 + 0.4 * (so < 0.8)), vis)
    return dx, dy, dz, vis, (fly_in & (t >= tin)) | (out_ & (so < 1))


KEY = None
REL0 = DS.REL
OUT_MIN = float(os.environ.get('OUT_MIN', '0'))          # out-transition only once the clip time passes this (absolute s)
ONLY = set(range(*[int(x) for x in os.environ['ONLY'].split('-')])) if os.environ.get('ONLY') else None
for fi in range(f0, f1):
    if ONLY is not None and fi not in ONLY and (KEY is not None or MODE != 'trans'):
        continue
    tabs = fi / FPS
    t = tabs % T                                           # the loop's own time (seamless wrap)
    rw, tau = wob(tabs)
    ampw = 1 + (WOB_A - 1) * rw
    DS.REL = REL0 * ampw
    off = GL * t
    a = np.pi * t / (2 * T)                                       # glide crossfade over one loop: seamless at T
    tmw = (t + (tau - tabs)) % (T * 1000)
    h1, d1 = surface(t, off, tmw); h2, d2 = surface(t, off - D, tmw)
    h = np.cos(a) * h1 + np.sin(a) * h2; det = np.cos(a) * d1 + np.sin(a) * d2
    h = (h + 2.6) / 5.2                                            # fixed range (no per-frame pumping)
    h = np.clip(h, 0, 1); h = h * h * (3 - 2 * h)
    hd = np.clip(h + DA * det, 0, 1)
    # ---- light ----
    px_x = SX / H; px_y = SY / H
    hs = cv2.GaussianBlur(hd, (0, 0), 0.8) * REL_ * ampw
    gy_, gx_ = np.gradient(hs)
    n = np.stack([-gx_ / px_x, gy_ / px_y, np.ones_like(gx_)], -1); n /= np.linalg.norm(n, axis=-1, keepdims=True)
    Lb = nv(0.25 + 0.15 * np.sin(2 * np.pi * t / T), 0.95, 0.35)
    lam = np.clip(n @ Lk, 0, 1)
    spec = np.clip(n @ nv(*(Lb + Vc)), 0, 1) ** float(os.environ.get('SPEC_P', '28'))
    spk = np.clip(n @ nv(*(Lk + Vc)), 0, 1) ** 60
    lum = float(os.environ.get('FILL', '0.12')) + float(os.environ.get('KEY', '0.42')) * lam ** 1.25 \
        + float(os.environ.get('SHEEN', '0.7')) * spec + 0.35 * spk
    cav = hs - cv2.GaussianBlur(hs, (0, 0), 25); lum = (lum * np.clip(1 + cav / (REL_ * 0.25 + 1e-6) * 0.6, 0.55, 1.3)).astype(np.float32)
    LN = float(os.environ.get('LUMNORM', '0'))                     # steady exposure: highlights keep a constant weight
    if LN > 0:
        q = np.percentile(lum[::4, ::4], 92); lum = (lum * (LN / max(q, 1e-3)) ** float(os.environ.get('LUMNORM_K', '0.85'))).astype(np.float32)
    nrow = np.floor(off / P + 1e-6); sh = float(off - nrow * P)
    pts = (gx0, (gy0 + sh).astype(np.float32))                  # float32: the samplers need it
    gid_row = row0 - int(nrow)                       # ground-row identity (constant as rows glide)
    vis = None
    DS.LANDOFF.clear()
    if MODE == 'trans':
        if KEY is None:                                           # probe: screen row of each dot -> near-first key
            os.environ['DBGDOTS'] = f'/tmp/probe_{f0}.npz'
            DS.render(hd.astype(np.float32), 1, t, pts, None, (col_id * 1_000_003 + (gid_row + 100_000)).astype(np.int64), None, lum)
            del os.environ['DBGDOTS']
            z_ = np.load(f'/tmp/probe_{f0}.npz')
            vv = z_['v']; onsc = (vv > 0) & (vv < H)
            vlo, vhi = np.percentile(vv[onsc], 1), np.percentile(vv[onsc], 99)
            KEY = np.clip((vhi - vv) / (vhi - vlo), 0, 1)        # 0 = bottom (near) .. 1 = horizon
            KEY = np.where(vv > H, 0.0, np.where(vv < 0, 1.0, KEY))
        Kb = 8
        tf = t
        if tabs < OUT_MIN:                                 # first pass: no dissolve yet
            tf = min(t, OUT0 - 0.01)
        if os.environ.get('NO_IN') and tf < OUT0 - 0.01:
            tf = max(tf, IN0 + ISP + IDU * 1.3 + 0.05)     # no forming (the clip starts mid-shot)
        sub = [flights(tf + (k + 0.5) / Kb * 0.5 / FPS, gid_row, KEY) for k in range(Kb)]
        moving = sub[0][4].any()
        if moving:                                                # motion blur: Kb positions over the open shutter
            pts = (np.tile(gx0, Kb), np.tile((gy0 + sh).astype(np.float32), Kb))
            DS.LANDOFF.update(dx=np.concatenate([s_[0] for s_ in sub]), dy=np.concatenate([s_[1] for s_ in sub]),
                              dz=np.concatenate([s_[2] for s_ in sub]))
            vis = np.concatenate([s_[3] for s_ in sub]) / Kb
        else:
            dx, dy, dz, vis, _ = sub[0]
            DS.LANDOFF.update(dx=dx, dy=dy, dz=dz)
    if os.environ.get('GAINFILE'):                                 # per-frame exposure levelling (loop-smoothed)
        os.environ['GAIN'] = str(float(os.environ['GAIN0']) * float(np.load(os.environ['GAINFILE'])[fi % 375]))
    gids = (col_id * 1_000_003 + (gid_row + 100_000)).astype(np.int64)  # twinkles ride with the ground row
    if pts[0].size != gids.size:
        gids = np.tile(gids, pts[0].size // gids.size)
    img = DS.render(hd.astype(np.float32), 1, t, pts, vis, gids, None, lum)
    cv2.imwrite(f'{outdir}/f{fi:04d}.png', (np.clip(img, 0, 1) * 65535 + 0.5).astype(np.uint16))
    print(fi, flush=True)
