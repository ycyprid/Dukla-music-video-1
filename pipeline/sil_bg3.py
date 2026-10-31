"""ONE dot sheet (landscape renderer, LAND mode): a strongly curved sheet behind the dancers whose brightness forms a
grid of soft circles; the dancers are CARVED out of the same dots (no pasted matte), and the dots along their edge
catch the backlight (value lift + twinkle gate) -> the outline is made of the sheet's own dots.
usage: python3 sil_bg3.py src.mp4 frame out.png"""
import sys, os
import numpy as np, cv2
import depth_sheet as DS
from common import W, H
import sil_bg

EV = lambda k, d: float(os.environ.get(k, d))
OW, OH = int(round(W * DS.OUTS)), int(round(H * DS.OUTS))
_C = {}


def nearness(gx, gy, t):
    """how far each sheet point comes towards the camera (0..1): right side near, its top / bottom nearer still"""
    br = 1 + EV('CV_BR', '0.035') * bar_pulse(t)                     # the wall swells on each bar's downbeat
    sx = np.clip(gx / W + EV('CV_SX0', '0.0'), 0, None); sy = (gy / H - EV('CV_YC', '0.5')) * 2
    n = (EV('CA', '1.0') * sx ** 2 + EV('CB', '1.2') * sx ** 2 * sy ** 2 + EV('CC', '0.25') * sy ** 2) * br
    return n / EV('CV_N', '2.3')


BT = 60.0 / EV('BPM', '146')


def _kick(ph, att, rel):
    a = np.clip(ph / att, 0, 1); a = a * a * (3 - 2 * a)
    return a * np.exp(-np.maximum(ph - att, 0) / rel)


def bar_pulse(t):
    return float(_kick((t + EV('BEAT_OFF', '0')) % (4 * BT), 0.10, 0.55))


def beat_pulse(t):
    return float(_kick((t + EV('BEAT_OFF', '0')) % BT, 0.06, 0.18))


def depmap(gx, gy, t):
    return 0.15 + 0.85 * np.clip(nearness(gx, gy, t), 0, 1)


def setup_env():
    os.environ.update(LAND=os.environ.get('SH_LAND', '1.5,1.5'), LAND_Y0=os.environ.get('SH_Y0', '0'),
                      LAND_V='0', LAND_FOG0='9', LAND_FOG='9', LAND_DENS=os.environ.get('SH_DENS', '0.6'),
                      LAND_ZREF=os.environ.get('SH_ZREF', '2.1'))
    DS.TILT, DS.YAW, DS.REL = EV('SH_TILT', '0.10'), EV('SH_YAW', '-0.15'), EV('SH_REL', '1.6')
    DS.DOFK = EV('SH_DOFK', '0.03')


def project(gx, gy, d):
    """the renderer's projection (LAND mode, no FIT / push), for any sheet points"""
    SX, SY = (float(x_) for x_ in os.environ['LAND'].split(','))
    X = (gx - W / 2) / H * SX; Y = -(gy - H / 2) / H * SY + float(os.environ['LAND_Y0']); Z = DS.Z0 - DS.REL * d
    ct, st, cy, sy = np.cos(DS.TILT), np.sin(DS.TILT), np.cos(DS.YAW), np.sin(DS.YAW)
    Zc = Z - DS.Z0 * 0.75; X1 = X * cy + Zc * sy; Z1 = -X * sy + Zc * cy
    Y2 = Y * ct - Z1 * st; Z2 = Y * st + Z1 * ct + DS.Z0 * 0.75
    return W / 2 + DS.FOC * X1 / Z2, H / 2 - DS.FOC * Y2 / Z2, Z2


def circles(gx, gy, t):
    """soft circles on a square grid in SHEET coords (so the sheet's own perspective / curve warps them), drifting"""
    S = EV('CI_CELL', '80')
    # far (left) -> near (right): one cell per bar
    x = gx / S - EV('CI_CPB', '1.0') * t / (4 * BT); y = gy / S + EV('CI_VY', '0.0') * t
    d = np.hypot(x - np.round(x), y - np.round(y))
    R = EV('CI_R', '0.36')
    e_ = np.clip((R - d) / EV('CI_EDGE', '0.07'), 0, 1); e_ = e_ * e_ * (3 - 2 * e_)
    return e_ * (1 - EV('CI_CORE', '0.30') * np.clip(d / R, 0, 1) ** 2)


def sample(img, u, v):
    n = u.size; pad = (-n) % 1024
    mx = np.pad(u.astype(np.float32), (0, pad)).reshape(-1, 1024); my = np.pad(v.astype(np.float32), (0, pad)).reshape(-1, 1024)
    return cv2.remap(img.astype(np.float32), mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).ravel()[:n]


def reframe(fr):
    if not os.environ.get('FOOT_CROP'):
        return fr
    cx_, cy_, cw_ = (float(v_) for v_ in os.environ['FOOT_CROP'].split(','))
    h0, w0 = fr.shape[:2]; sc = w0 / cw_
    M = np.array([[sc, 0, -cx_ * sc], [0, sc, -cy_ * sc]], np.float32)
    return cv2.warpAffine(fr, M, (w0, h0), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def _rand(n):
    if _C.get('rn') is None or _C['rn'][0].size != n:
        r = np.random.default_rng(2024); _C['rn'] = [r.random(n) for _ in range(8)]
    return _C['rn']


def flights(t, gx, gy, rimd):
    """per-dot world offsets (dx, dy, dz), visibility, any-in-flight.
    IN : every dot starts in a small knot at the frame centre (far back), bursts out in all directions on a gently
         curling path and settles on its place; the centre lands first, the burst spreads outwards, a few stragglers
         lag; the outline dots land last.
    OUT: dots peel off and ride the background's own flow (along the sheet, left -> right, round the curve towards
         the camera), accelerating and fading; the far left goes first, the wave travels right; outline last."""
    n = gx.size
    r1, r2, r3, r4, r5, r6, r7, r8 = _rand(n)
    SX, SY = (float(x_) for x_ in os.environ['LAND'].split(','))
    Y0 = float(os.environ['LAND_Y0'])
    X = (gx - W / 2) / H * SX; Y = -(gy - H / 2) / H * SY + Y0; Z = DS.Z0 - DS.REL * depmap(gx, gy, t)
    dx = np.zeros(n); dy = np.zeros(n); dz = np.zeros(n); vis = np.ones(n); fly = False
    TLc = EV('CLIPLEN', '8.6'); BTt = 60.0 / 146
    # ---------------- in ----------------
    isp, idu = EV('IN_SP', str(2.0 * BTt)), EV('IN_DU', str(2.6 * BTt))
    rad = np.hypot(gx / W - 0.5, (gy / H - 0.5) * H / W) / 0.6
    lag = (r7 < EV('LAG_FR', '0.16')) * (0.15 + 0.25 * r8)
    key = np.clip(0.62 * rad + 0.38 * r1, 0, 1)
    tin = 0.04 + isp * key + lag + EV('RIM_LATE', '0.18') * np.clip(rimd * 2, 0, 1)
    din = idu * (0.8 + 0.4 * r2)
    s = np.clip((t - tin) / din, 0, 1)
    if (s < 1).any():
        fly = True
        e = 0.82 * (1 - (1 - s) ** 2.3) + 0.18 * DS._back_out(s, 0.6)
        if 'knot' not in _C:                                          # world point that is seen at the frame centre
            zk = DS.Z0 - DS.REL * 0.3 + EV('IN_DEPTH', '0.9')
            ax, ay = np.meshgrid(np.linspace(-1.5, 1.5, 301), np.linspace(-1.5, 1.5, 301))
            ct, st, cy, sy = np.cos(DS.TILT), np.sin(DS.TILT), np.cos(DS.YAW), np.sin(DS.YAW)
            Zc = zk - DS.Z0 * 0.75; X1 = ax * cy + Zc * sy; Z1 = -ax * sy + Zc * cy
            Y2 = ay * ct - Z1 * st; Z2 = ay * st + Z1 * ct + DS.Z0 * 0.75
            uu = W / 2 + DS.FOC * X1 / Z2; vv = H / 2 - DS.FOC * Y2 / Z2
            k_ = np.argmin(np.hypot(uu - W * EV('KNOT_X', '0.5'), vv - H * EV('KNOT_Y', '0.5')))
            _C['knot'] = (ax.ravel()[k_], ay.ravel()[k_])
        th = 2 * np.pi * r3; rr = EV('KNOT_R', '0.05') * np.sqrt(r4)
        Xs = _C['knot'][0] + rr * np.cos(th); Ys = _C['knot'][1] + rr * np.sin(th) * 0.8       # the knot at the centre
        # curl: sideways to the burst direction, strongest mid-flight
        ux, uy = X - Xs, Y - Ys; ln = np.hypot(ux, uy) + 1e-6
        cu = EV('CURL', '0.22') * (r5 - 0.5) * np.sin(np.pi * np.clip(e, 0, 1)) * ln
        dx = (Xs - X) * (1 - e) + (-uy / ln) * cu
        dy = (Ys - Y) * (1 - e) + (ux / ln) * cu
        dz = -(DS.Z0 - DS.REL * 0.3 + EV('IN_DEPTH', '0.9') - Z) * (1 - e) + EV('IN_ARC', '0.35') * r6 * np.sin(np.pi * np.clip(e, 0, 1))
        vis = np.where(t < tin, 0.0, np.clip(s / 0.05, 0, 1)) * (1 + 0.6 * np.sin(np.pi * s) * (s < 1))
    # ---------------- out ----------------
    o0 = TLc - EV('OUT_LEAD', '1.45'); osp = EV('OUT_SP', '0.62'); odu = EV('OUT_DU', '0.55')
    keyo = np.clip(0.60 * np.clip(gx / W, 0, 1) + 0.25 * r1 + 0.15 * np.clip(rimd * 2, 0, 1), 0, 1)
    tout = o0 + osp * keyo + (r7 < EV('LAG_FR', '0.16')) * 0.12 * r8
    dout = odu * (0.8 + 0.4 * r2)
    so = np.clip((t - tout) / dout, 0, 1)
    if (so > 0).any():
        fly = True
        eo = so ** EV('OUT_EP', '1.6')
        Dg = W * (EV('OUT_D0', '0.35') + EV('OUT_D1', '0.55') * r3) * eo                                   # along the sheet, with the flow
        X2 = ((gx + Dg) - W / 2) / H * SX; Z2 = DS.Z0 - DS.REL * np.clip(depmap(gx + Dg, gy, t), 0, 1.25)
        o = so > 0
        dx = np.where(o, X2 - X, dx)
        dz = np.where(o, np.clip(Z - Z2, 0, None) * EV('OUT_CURVE', '0.35') + EV('OUT_LIFT', '0.12') * r6 * eo, dz)
        dy = np.where(o, 0.10 * (r5 - 0.5) * np.sin(np.pi * so) + 0.06 * eo * (r4 - 0.3), dy)
        vo = np.clip((1 - so) / EV('OUT_FADE', '0.85'), 0, 1)
        vis = np.where(o, vis * vo * vo * (3 - 2 * vo) * (1 + EV('OUT_GLOW', '1.2') * np.sin(np.pi * so)), vis)
    return dx, dy, dz, vis, fly


def frame(fr, t, state=None):
    fr = reframe(fr)
    F = sil_bg.footage(fr, state)                       # matte mF (output res, screen = reframed footage), light
    setup_env()
    P = EV('SH_PITCH', '4.0')
    if 'lat' not in _C:
        xs = np.arange(-0.15 * W, 1.15 * W, P); ys = np.arange(-0.15 * H, 1.15 * H, P)
        gx, gy = np.meshgrid(xs, ys)
        gx, gy = gx.ravel().astype(np.float32), gy.ravel().astype(np.float32)
        DS.NBASE['n'] = gx.size
        if EV('FINE_EDGE', '1') > 0:            # in-between dots (cell centres): only ever lit along the outline
            gx = np.concatenate([gx, gx + P / 2]); gy = np.concatenate([gy, gy + P / 2])
        _C['lat'] = (gx, gy)
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32); _C['pix'] = (xx, yy)
    gx, gy = _C['lat']; xx, yy = _C['pix']
    dep = depmap(xx, yy, t).astype(np.float32)
    # where every dot lands (exactly as the renderer will put it)
    u, v, z = project(gx, gy, depmap(gx, gy, t))
    S = DS.OUTS
    mF = F['mF']
    m_d = sample(mF, u * S, v * S)
    # the backlit edge: a band just OUTSIDE the silhouette, a dot or two wide, brightest where the light behind is
    bw = EV('RIM_W', '6.0')
    band = np.clip(cv2.GaussianBlur(mF, (0, 0), bw) * EV('RIM_G', '2.4'), 0, 1) * (1 - mF)
    band = np.clip(band, 0, 1) ** EV('RIM_P', '1.3')
    b_d = sample(band, u * S, v * S)
    light = cv2.resize(F['light'], (W, H), interpolation=cv2.INTER_AREA)
    if state is not None:                         # the backlight changes slowly: no frame-to-frame pumping
        k_ = EV('LIGHT_EMA', '0.06')
        state['lightS'] = light if 'lightS' not in state else state['lightS'] + (light - state['lightS']) * k_
        light = state['lightS']
    l_d = np.clip(sample(light, u, v), 0, 1)
    # sheet brightness: the circles x the backlight (looked up where the sheet point is seen)
    us, vs, _ = project(xx, yy, dep)
    lt = np.clip(sample(light, us.ravel(), vs.ravel()).reshape(H, W), 0, 1) ** EV('SH_LIGHTG', '0.9')
    lum = lt * (EV('CI_FLOOR', '0.03') + (1 - EV('CI_FLOOR', '0.03')) * circles(xx, yy, t)) * EV('SH_LUMK', '2.0')
    lum = lum * (1 + EV('BEAT_LUM', '0.025') * beat_pulse(t) + EV('BAR_LUM', '0.03') * bar_pulse(t))
    # carve the dancers out of the sheet (anti-aliased: edge dots fade with their coverage)
    cut = np.clip((m_d - 0.15) / 0.55, 0, 1); cut = cut * cut * (3 - 2 * cut)
    vis = (1 - cut).astype(np.float32)
    lift = (b_d * EV('RIM_V', '0.85') * (0.45 + 0.55 * l_d ** 0.6)).astype(np.float32)
    gate = (b_d * EV('RIM_F', '1.0')).astype(np.float32)
    nb = DS.NBASE.get('n', u.size)
    if u.size > nb:
        # fine edge dots: small and darker, tracing the contour between the main dots, both sides of it
        e1 = np.clip(cv2.GaussianBlur(mF, (0, 0), EV('FE_W', '5.0')), 0, 1)
        eb = np.clip(4 * e1 * (1 - e1), 0, 1) ** EV('FE_P', '0.7')            # 1 right at the edge, 0 away from it
        eb_d = sample(eb, u[nb:] * S, v[nb:] * S)
        fvis = eb_d * (1 - np.clip((m_d[nb:] - 0.35) / 0.4, 0, 1)) * EV('FE_VIS', '0.85')
        vis = np.concatenate([vis[:nb], fvis.astype(np.float32)])
        lift = np.concatenate([lift[:nb], (b_d[nb:] * EV('FE_LIFT', '0.25') - EV('FE_DARK', '0.18')).astype(np.float32)])
        gate = np.concatenate([gate[:nb], (b_d[nb:] * EV('FE_F', '0.6')).astype(np.float32)])
    DS.DOTADD['v'] = lift
    DS.DOTADD['f'] = gate
    os.environ['GAIN'] = os.environ.get('SH_GAIN', '2.0')
    DS.LANDOFF.clear()
    if os.environ.get('DTRANS'):
        Kb = int(EV('MB_K', '5'))
        sub = [flights(t + (k + 0.5) / Kb * 0.5 / 25, gx, gy, b_d) for k in range(Kb)]
        if any(s_[4] for s_ in sub):                       # in flight: motion blur over the open shutter
            pts = (np.tile(gx, Kb), np.tile(gy, Kb))
            DS.LANDOFF.update(dx=np.concatenate([s_[0] for s_ in sub]), dy=np.concatenate([s_[1] for s_ in sub]),
                              dz=np.concatenate([s_[2] for s_ in sub]))
            visk = np.concatenate([vis * s_[3] for s_ in sub]) / Kb
            for k_ in ('v', 'f'):
                DS.DOTADD[k_] = np.tile(DS.DOTADD[k_], Kb)
            img = DS.render(dep, 2, t, pts, visk.astype(np.float32), None, None, lum.astype(np.float32))
        else:
            img = DS.render(dep, 2, t, (gx, gy), (vis * sub[0][3]).astype(np.float32), None, None, lum.astype(np.float32))
    else:
        img = DS.render(dep, 2, t, (gx, gy), vis, None, None, lum.astype(np.float32))
    # glow that spilled over the silhouette from bright neighbours: keep a whisper of it (backlight wrap), no more
    inner = cv2.erode(mF, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    img = img * (1 - EV('INNER_DARK', '0.0') * inner[..., None])
    return img, dict(u=u, v=v, m=mF)


if __name__ == '__main__':
    src, fi, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    cap = cv2.VideoCapture(src)
    for _ in range(fi):
        cap.grab()
    ok, fr = cap.read()
    img, parts = frame(fr, fi / 25.0)
    if os.environ.get('CHECKPROJ'):
        os.environ['DBGDOTS'] = '/tmp/claude-0/-home-claude/f42362b3-e77e-5554-9434-580f66a1e90e/scratchpad/p3.npz'
        frame(fr, fi / 25.0)
        z = np.load(os.environ['DBGDOTS'])
        print('projection check: max |du| %.4f |dv| %.4f px' % (np.abs(z['u'] - parts['u']).max(), np.abs(z['v'] - parts['v']).max()))
        del os.environ['DBGDOTS']
    cv2.imwrite(out, (img * 65535 + 0.5).astype(np.uint16))
    print('done', out)
