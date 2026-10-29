"""Backlit silhouettes over a FINE-DOT backdrop whose brightness forms big soft circles (the compressed-thumbnail look).
  backdrop : the landscape renderer (LAND mode: tiny dots in perspective, halftone size, glints, fog) on a sheet
             leaning back behind the couple; its brightness = a lattice of soft circles painted ON the sheet
             (so perspective stretches them into ellipses) x the footage's backlight (sampled where each dot lands)
  rim      : the sparkling fine-dot outline (dot renderer, footage plane)
  matte    : the footage matte pushed through the rim renderer's own projection -> cut-out and outline coincide
usage: python3 sil_bg2.py src.mp4 frame out.png"""
import sys, os
import numpy as np, cv2
import depth_sheet as DS
from common import W, H
import sil_bg

EV = lambda k, d: float(os.environ.get(k, d))
OW, OH = int(round(W * DS.OUTS)), int(round(H * DS.OUTS))
SCR = os.environ.get('DBGDIR', '/tmp')
CAMERA = dict(rim=dict(TILT=DS.TILT, YAW=DS.YAW, REL=DS.REL), bg=None)
_C = {}


def setcam(which):
    if which == 'rim':
        DS.TILT, DS.YAW, DS.REL = CAMERA['rim']['TILT'], CAMERA['rim']['YAW'], CAMERA['rim']['REL']
        DS.DOFK = CAMERA.setdefault('dofk', DS.DOFK)
        for k in ('LAND', 'LAND_Y0', 'LAND_V', 'LAND_FOG0', 'LAND_FOG', 'LAND_DENS', 'LAND_ZREF', 'NOSPECKS', 'FABRIC'):
            os.environ.pop(k, None)
        os.environ['FABRIC'] = '1'
        os.environ['DARKCUT'] = os.environ.get('RIM_DARKCUT', '0.18,0.45')
        if CAMERA.get('edge'):
            os.environ['EDGE'] = CAMERA['edge']
        if CAMERA.get('fabk'):
            os.environ['FAB_K'] = CAMERA['fabk']
        os.environ.pop('FAB_HP', None)
        os.environ['FIT'] = 'cover'; os.environ.setdefault('FIT_D', '0.5'); os.environ.setdefault('FIT_OVER', '1.03')
    else:
        DS.TILT, DS.YAW, DS.REL = EV('BG_TILT', '0.75'), EV('BG_YAW', '0.0'), EV('BG_REL', '0.05')
        CAMERA.setdefault('dofk', DS.DOFK); DS.DOFK = EV('BG_DOFK', '0.02')     # no halftone/soft switch-over line
        os.environ.update(LAND=os.environ.get('BG_LAND', '4.2,4.2'), LAND_Y0=os.environ.get('BG_LY0', '0.6'),
                          LAND_V=os.environ.get('BG_LV', '0'), LAND_FOG0=os.environ.get('BG_FOG0', '3.2'),
                          LAND_FOG=os.environ.get('BG_FOG', '3.0'), LAND_DENS=os.environ.get('BG_DENS', '1.2'),
                          LAND_ZREF=os.environ.get('BG_ZREF', '1.6'), NOSPECKS='1')
        os.environ.pop('DARKCUT', None); os.environ.pop('FIT', None)
        CAMERA.setdefault('edge', os.environ.get('EDGE')); os.environ.pop('EDGE', None)      # the sheet continues past its image
        CAMERA.setdefault('fabk', os.environ.get('FAB_K', '6')); os.environ['FAB_K'] = os.environ.get('BG_FABK', '2.0')
        os.environ['FAB_HP'] = os.environ.get('BG_FABHP', '0.12')


def blobs(gx, gy, t):
    """soft circles on a hex lattice in sheet coords (px), slowly drifting / turning; 1 at a centre, 0 between"""
    S = EV('BB_S', '78')
    ph = np.radians(EV('BB_ROT', '-28') + EV('BB_ROTV', '1.2') * t)
    x = (gx - W / 2) / S; y = (gy - H / 2) / S
    x, y = x * np.cos(ph) - y * np.sin(ph), x * np.sin(ph) + y * np.cos(ph)
    x = x + EV('BB_VX', '0.06') * t; y = y * EV('BB_ASP', '1.0') + EV('BB_VY', '0.025') * t
    # hex lattice: nearest centre among the two offset rectangular sub-lattices
    r3 = np.sqrt(3.0)
    def nearest(xx, yy):
        cx = np.round(xx); cy = np.round(yy / r3) * r3
        return np.hypot(xx - cx, yy - cy)
    d = np.minimum(nearest(x, y), nearest(x - 0.5, y - r3 / 2))
    R = EV('BB_R', '0.40')
    if os.environ.get('BB_EDGE'):                                  # defined circle, soft edge, hotter centre
        e_ = np.clip((R - d) / EV('BB_EDGE', '0.12'), 0, 1); e_ = e_ * e_ * (3 - 2 * e_)
        b = e_ * (1 - EV('BB_CORE', '0.35') * np.clip(d / R, 0, 1) ** 2)
    else:
        b = np.clip(1 - (d / R) ** 2, 0, 1) ** EV('BB_P', '1.3')
    return b


def blobs_warp(x, y, t):
    """the circle grid laid out in SCREEN space on a warped grid (the hand-drawn reference): cells grow towards the
    right (perspective), columns lean on the left and bow outwards top / bottom on the right, rows fan out to the
    right. x, y: screen position 0..1. Slow breathing + drift."""
    x = np.clip(x, -0.5, 1.5); y = np.clip(y, -0.5, 1.5)
    yc = EV('GW_YC', '0.50')
    yn = (y - yc) * H / W                                            # same units as x
    k = EV('GW_K', '0.75') * (1 + EV('GW_KB', '0.08') * np.sin(2 * np.pi * t / EV('GW_TB', '6.0')))
    bend = EV('GW_BEND', '0.45') * (1 + EV('GW_BB', '0.12') * np.sin(2 * np.pi * t / EV('GW_TB', '6.0') + 1.3))
    xp = x - bend * yn * yn * np.clip(x, 0, None) ** 2 * 4 + EV('GW_LEAN', '0.10') * yn * (1 - x)
    s_ = np.exp(k * xp)
    c0 = EV('GW_CELL', '0.060')
    gx = (1 - np.exp(-k * xp)) / k / c0 + EV('GW_VX', '0.12') * t
    gy = yn / (c0 * s_) * (1 + EV('GW_FAN', '0.25') * np.clip(x, 0, 1) * np.abs(yn) * 4) + EV('GW_VY', '0.0') * t
    d = np.hypot(gx - np.round(gx), gy - np.round(gy))
    R = EV('BB_R', '0.36')
    e_ = np.clip((R - d) / EV('BB_EDGE', '0.06'), 0, 1); e_ = e_ * e_ * (3 - 2 * e_)
    return e_ * (1 - EV('BB_CORE', '0.35') * np.clip(d / R, 0, 1) ** 2)


def bg_layer(t, light_scr, m_scr):
    setcam('bg')
    P = EV('BG_PITCH', '1.6')
    if 'bgl' not in _C:
        PY = EV('BG_PITCHY', str(P))
        xs = np.arange(-0.1 * W, 1.1 * W, P); ys = np.arange(-0.1 * H, 1.1 * H, PY)
        gxs, gys = np.meshgrid(xs, ys)
        _C['bgl'] = (gxs.ravel().astype(np.float32), gys.ravel().astype(np.float32), gxs.shape, xs, ys)
    gx, gy, shp, xs, ys = _C['bgl']
    dep = np.full((H, W), 0.5, np.float32)
    if 'bguv' not in _C:                                 # probe once: where each sheet point lands on screen
        os.environ['DBGDOTS'] = f'{SCR}/bgprobe.npz'
        DS.render(dep, 1, t, (gx, gy), None, None, None, np.zeros((H, W), np.float32))
        del os.environ['DBGDOTS']
        z = np.load(f'{SCR}/bgprobe.npz')
        U = z['u'].reshape(shp).astype(np.float32); V = z['v'].reshape(shp).astype(np.float32)
        # sheet-pixel maps of the screen position (for the footage light, sampled where the dot lands)
        mx, my = np.meshgrid((np.arange(W) - xs[0]) / (xs[1] - xs[0]), (np.arange(H) - ys[0]) / (ys[1] - ys[0]))
        _C['bguv'] = (cv2.remap(U, mx.astype(np.float32), my.astype(np.float32), cv2.INTER_LINEAR),
                      cv2.remap(V, mx.astype(np.float32), my.astype(np.float32), cv2.INTER_LINEAR))
    Us, Vs = _C['bguv']
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    b = blobs_warp(Us / W, Vs / H, t) if os.environ.get('GRIDWARP') else blobs(xx, yy, t)
    lt = cv2.remap(light_scr, Us, Vs, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    lt = np.clip(lt, 0, 1) ** EV('BG_LIGHTG', '0.55')                 # lift the dim side: circles read across the frame
    # gentle relief so the sheen slides over the sheet (crests of a slow swell)
    sw = 0.5 + 0.5 * np.sin((xx / W * 2.1 + yy / H * 1.3) * 2 * np.pi * 0.5 - 0.6 * t)
    dep = (0.45 + 0.10 * sw).astype(np.float32)
    lum = lt * (EV('BB_FLOOR', '0.06') + (1 - EV('BB_FLOOR', '0.06')) * b) * EV('BG_LUMK', '1.15')
    lum = lum * (0.85 + 0.3 * sw)
    os.environ['GAIN'] = os.environ.get('BG_GAIN', '2.0')
    img = DS.render(dep, 2, t, (gx, gy), None, None, None, lum.astype(np.float32))
    return img


def rim_layer(t, F):
    setcam('rim')
    os.environ['GAIN'] = os.environ.get('RIM_GAIN', '1.4')
    dep = np.full((H, W), 0.5, np.float32)                # constant depth: the projection is one exact plane map
    if 'rimpts' not in _C:
        p0 = DS.PITCH_PX; DS.PITCH_PX = EV('RIM_PITCH', str(p0))      # finer outline dots
        _C['rimpts'] = DS.lattice(); DS.PITCH_PX = p0
        os.environ['DBGDOTS'] = f'{SCR}/rimprobe.npz'
        DS.render(dep, 1, t, _C['rimpts'], None, None, None, np.zeros((H, W), np.float32))
        del os.environ['DBGDOTS']
        z = np.load(f'{SCR}/rimprobe.npz')
        src = np.stack(_C['rimpts'], -1).astype(np.float32); dst = np.stack([z['u'], z['v']], -1).astype(np.float32)
        ins = (src[:, 0] > 2) & (src[:, 0] < W - 2) & (src[:, 1] > 2) & (src[:, 1] < H - 2)
        src, dst = src[ins], dst[ins]
        Hm, _ = cv2.findHomography(src, dst, 0)
        Sc = np.diag([DS.OUTS, DS.OUTS, 1.0])
        _C['Hm'] = Sc @ Hm @ np.linalg.inv(Sc)                    # output-res footage -> output-res screen
        err = np.abs(cv2.perspectiveTransform(src[None], Hm)[0] - dst).max()
        print('rim projection = plane map, max error %.3f px' % err, flush=True)
    img = DS.render(dep, 1, t, _C['rimpts'], None, None, None, F['rim_lum'].astype(np.float32))
    return img


_NZ = {}
def _noise(seed, scale):
    if (seed, scale) not in _NZ:
        z = np.random.default_rng(seed).random((OH // 4 + 2, OW // 4 + 2)).astype(np.float32)
        z = cv2.GaussianBlur(z, (0, 0), scale / 4); z = (z - z.min()) / (np.ptp(z) + 1e-6)
        _NZ[(seed, scale)] = cv2.resize(z, (OW, OH), interpolation=cv2.INTER_CUBIC)
    return _NZ[(seed, scale)]


def glitter(mS, light_out, t):
    """sparkles strung along the silhouette contour (over it): evenly spaced points with a little scatter, each
    twinkling on its own rhythm (rhythm field fixed in space, so glints ride along as the edge moves); brightness
    follows the light behind; a few bigger flares"""
    bw = (mS > 0.5).astype(np.uint8)
    cs, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    step = EV('GL_STEP', '5.0'); pts = []
    for c in cs:
        c = c[:, 0, :].astype(np.float32)
        if len(c) < 12:
            continue
        seg = np.hypot(*np.diff(np.vstack([c, c[:1]]), axis=0).T); arc = np.concatenate([[0], np.cumsum(seg)])[:-1]
        sm = np.arange(0, arc[-1], step)
        pts.append(np.stack([np.interp(sm, arc, c[:, 0]), np.interp(sm, arc, c[:, 1])], -1))
    img = np.zeros((OH, OW, 3), np.float32)
    if not pts:
        return img
    P = np.concatenate(pts)
    # a second, looser band of sparkles scattered across the edge (mostly onto the silhouette), along the normal
    SC = EV('GL_SCAT', '1.0')
    if SC > 0:
        mb = cv2.GaussianBlur(mS, (0, 0), 3); gy_, gx_ = np.gradient(mb)
        pi_ = np.clip(P.astype(int), [0, 0], [OW - 1, OH - 1])
        nx = gx_[pi_[:, 1], pi_[:, 0]]; ny = gy_[pi_[:, 1], pi_[:, 0]]; nn = np.hypot(nx, ny) + 1e-6
        nx, ny = nx / nn, ny / nn                                          # points INTO the silhouette
        rr = np.random.default_rng(3)
        pick = rr.random(len(P)) < EV('GL_SCATD', '0.45') * SC
        off = (rr.normal(EV('GL_SCATM', '2.5'), EV('GL_SCATS', '3.5'), len(P)))[pick]
        Q = P[pick] + np.stack([nx[pick] * off, ny[pick] * off], -1)
        P = np.concatenate([P, Q])
    xi = np.clip(P[:, 0].astype(int), 0, OW - 1); yi = np.clip(P[:, 1].astype(int), 0, OH - 1)
    jx = (_noise(11, 9)[yi, xi] - 0.5) * EV('GL_JIT', '3.0'); jy = (_noise(12, 9)[yi, xi] - 0.5) * EV('GL_JIT', '3.0')
    ph = _noise(13, 40)[yi, xi] * 6.0; fq = 0.6 + 1.6 * _noise(14, 60)[yi, xi]
    tw = np.maximum(np.sin(2 * np.pi * (fq * t + ph)), 0) ** 6
    lt = np.clip(light_out[yi, xi], 0, 1)
    wv = (0.35 + 0.65 * lt) * (0.25 + 1.6 * tw) * EV('GL_K', '1.0')
    big = (_noise(15, 30)[yi, xi] > 0.72) & (tw > 0.6)
    col = DS.L.PAL[(np.clip(0.55 + 0.45 * lt + 0.25 * tw, 0, 1) * 255).astype(np.uint8)]
    acc = np.zeros((OH, OW, 3), np.float32); accb = np.zeros((OH, OW, 3), np.float32)
    X = np.clip(P[:, 0] + jx, 0, OW - 1); Y = np.clip(P[:, 1] + jy, 0, OH - 1)
    Xi, Yi = X.astype(int), Y.astype(int)
    for c_ in range(3):
        np.add.at(acc[..., c_], (Yi, Xi), wv * col[:, c_])
        np.add.at(accb[..., c_], (Yi[big], Xi[big]), wv[big] * col[big, c_])
    # crisp little discs, three sizes (own per position), brighter ones a touch bigger
    sz = _noise(16, 14)[yi, xi] + 0.35 * tw
    core = np.zeros_like(acc)
    R0 = EV('GL_R', '1.6')
    for lo_, hi_, rr in ((-1, 0.45, R0 * 0.75), (0.45, 0.75, R0), (0.75, 9, R0 * 1.35)):
        q = (sz >= lo_) & (sz < hi_)
        if not q.any():
            continue
        a_ = np.zeros_like(acc)
        for c_ in range(3):
            np.add.at(a_[..., c_], (Yi[q], Xi[q]), wv[q] * col[q, c_])
        k_ = DS.L.disc(rr); k_ = k_ / k_.max()
        core += cv2.filter2D(a_, -1, k_)
    core = cv2.GaussianBlur(core, (0, 0), 0.5) * EV('GL_CORE', '1.6')
    halo = cv2.GaussianBlur(acc, (0, 0), 3.0) * 4 + cv2.GaussianBlur(accb, (0, 0), 7.0) * 18
    # tiny cross flare on the biggest glints
    fl = cv2.GaussianBlur(accb, (0, 0), 0.8)
    kx = cv2.getGaussianKernel(41, 9).T; ky = cv2.getGaussianKernel(41, 9)
    flare = (cv2.filter2D(fl, -1, kx * cv2.getGaussianKernel(41, 0.8).max()) + cv2.filter2D(fl, -1, ky * cv2.getGaussianKernel(41, 0.8).max())) * 3
    return core + halo * EV('GL_HALO', '1.0') + flare * EV('GL_FLARE', '1.0')


def reframe(fr, i=None):
    """punch in on the dancers: FOOT_CROP=x,y,w (16:9 window in source px), sub-pixel, back to the source size"""
    if not os.environ.get('FOOT_CROP'):
        return fr
    cx_, cy_, cw_ = (float(v_) for v_ in os.environ['FOOT_CROP'].split(','))
    h0, w0 = fr.shape[:2]; sc = w0 / cw_
    M = np.array([[sc, 0, -cx_ * sc], [0, sc, -cy_ * sc]], np.float32)
    return cv2.warpAffine(fr, M, (w0, h0), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def compose(fr, t, state=None):
    fr = reframe(fr)
    F = sil_bg.footage(fr, state)
    setcam('rim')
    rim = rim_layer(t, F)
    if EV('RIM_LAT', '1') == 0:                 # lattice outline off: the contour glitter carries the edge
        rim = rim * 0
    mS = cv2.warpPerspective(F['mF'], _C['Hm'], (OW, OH), flags=cv2.INTER_LINEAR, borderValue=0)   # matte where the outline is
    light_d = cv2.resize(F['light'], (W, H), interpolation=cv2.INTER_AREA)
    # the light map lives in footage coords: put it on the screen the same way
    light_s = cv2.warpPerspective(light_d, np.diag([1 / DS.OUTS, 1 / DS.OUTS, 1]) @ _C['Hm'] @ np.diag([DS.OUTS, DS.OUTS, 1]), (W, H), borderMode=cv2.BORDER_REPLICATE)
    bg = bg_layer(t, light_s, mS)
    m3 = mS[..., None]
    inner = cv2.erode(mS, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))[..., None]
    bgm = bg * (1 - m3)
    wrap = cv2.GaussianBlur(bg, (0, 0), 5) * m3 * cv2.GaussianBlur(1 - mS, (0, 0), 3)[..., None] * EV('WRAP', '0.30')
    rimm = rim * (1 - inner * EV('RIM_INNER_KILL', '0.0'))
    base = bgm + wrap
    out = 1 - (1 - np.clip(base, 0, 1)) * (1 - np.clip(rimm, 0, 1))               # screen blend
    if EV('GL_K', '1.0') > 0:
        lo = cv2.resize(light_s, (OW, OH), interpolation=cv2.INTER_CUBIC)
        g_ = glitter(mS, lo, t)
        out = 1 - (1 - out) * (1 - np.clip(g_ / (1 + g_), 0, 1))
    return out, dict(bg=bg, rim=rim, m=mS)


if __name__ == '__main__':
    src, fi, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    cap = cv2.VideoCapture(src)
    for _ in range(fi):
        cap.grab()
    ok, fr = cap.read()
    img, parts = compose(fr, fi / 25.0)
    cv2.imwrite(out, (img * 65535 + 0.5).astype(np.uint16))
    cv2.imwrite(out.replace('.png', '_bg.jpg'), (np.clip(parts['bg'], 0, 1) * 255).astype(np.uint8))
    # alignment check: rim dots (green) over the matte (red)
    chk = np.zeros((OH, OW, 3), np.float32); chk[..., 2] = parts['m'] * 0.6; chk[..., 1] = np.clip(parts['rim'].max(-1) * 2, 0, 1)
    cv2.imwrite(out.replace('.png', '_align.jpg'), (chk * 255).astype(np.uint8))
    print('done', out)
