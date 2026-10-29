"""Backlit silhouette over a procedural BIG-DOT sheet (the warped-circle look).
  background : an undulating sheet of large dots tilted back behind the subjects; every dot is drawn as the
               ellipse the surface makes of it (perspective + wave tilt), lit like the landscape loop
               (key light, sliding sheen, cavity), brightness / colour led by the footage's own backlight
  subjects   : solid black (smooth full-res matte), faint backlight wrap inside the edge
  rim        : the fine sparkling dot outline from the backlit look (depth_sheet renderer, rim only)
usage: python3 sil_bg.py src.mp4 frame out.png        (stills; motion uses the same functions)"""
import sys, os
import numpy as np, cv2
import depth_sheet as DS
import sheet_loop2 as L
from common import W, H

EV = lambda k, d: float(os.environ.get(k, d))
OW, OH = int(round(W * DS.OUTS)), int(round(H * DS.OUTS))


# ------------------------------------------------------------------ footage: matte, rim, backlight map
def footage(fr, state=None):
    """fr: full-res BGR. Returns design-res rim/lum/dep for the rim layer, full-res matte, light map (output res)."""
    st = state if state is not None else {}
    sm_ = lambda k, v, a=0.15: st.__setitem__(k, v if k not in st else st[k] + (v - st[k]) * a) or st[k]
    frw = cv2.resize(fr, (W, H), interpolation=cv2.INTER_AREA)
    g = cv2.cvtColor(frw, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
    gb = cv2.GaussianBlur(g, (0, 0), 1.0)
    sm = cv2.resize(gb, (192, 108), interpolation=cv2.INTER_AREA)
    det = np.abs(sm - cv2.GaussianBlur(sm, (0, 0), 2))
    cand = (sm > np.percentile(sm, 45)) & (det < 0.04)
    yy, xx = np.mgrid[0:108, 0:192].astype(np.float32); xx /= 192; yy /= 108
    Bs = np.stack([np.ones_like(xx), xx, yy, xx * xx, xx * yy, yy * yy], -1)
    coef = np.linalg.lstsq(Bs[cand], sm[cand], rcond=None)[0]
    skys = sm_('sky', np.clip(Bs @ coef, 0.6 * np.median(sm[cand]), 1.0).astype(np.float32))
    sky = cv2.resize(skys, (W, H), interpolation=cv2.INTER_CUBIC)
    T = EV('BL_T', '0.55')
    m = np.clip((sky * T - gb) / (0.10 * sky + 0.01), 0, 1)
    m = cv2.GaussianBlur(cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)), (0, 0), 0.8)
    out_ = cv2.GaussianBlur(1 - m, (0, 0), EV('BL_RIMW', '2.2'))
    rim = np.clip(m * out_ * 2.2, 0, 1) ** 0.8
    rim = rim * np.clip(sky / (np.percentile(sky, 90) + 1e-6), 0.25, 1) ** 0.6
    # full-res matte (smooth edge for the black cut-out)
    gF = cv2.GaussianBlur(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255, (0, 0), 1.2)
    skyF = cv2.resize(skys, (fr.shape[1], fr.shape[0]), interpolation=cv2.INTER_CUBIC)
    mF = np.clip((skyF * T - gF) / (0.10 * skyF + 0.01), 0, 1)
    mF = cv2.GaussianBlur(cv2.morphologyEx(mF, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)), (0, 0), 1.0)
    mF = cv2.resize(mF, (OW, OH), interpolation=cv2.INTER_AREA)
    mF = np.clip(mF * EV('MATTE_HARD', '1.7'), 0, 1)          # motion-blurred limbs stay solid
    # backlight map: the sky's own brightness (smooth model, figures removed) + bloom around the figures
    fg = (m > 0.5).astype(np.uint8); fg[int(H * EV('BL_GROUND', '1.0')):] = 0
    dt = cv2.distanceTransform(1 - fg, cv2.DIST_L2, 5) / H
    aura = np.exp(-dt / EV('BL_AURA', '0.35'))
    AF = EV('BL_FLOOR', '0.45')
    skn = np.clip(sky / sm_('skyref', float(np.percentile(sky, 99.5)), 0.1), 0, 1) ** EV('BG_SKYG', '1.6')
    light = skn * (AF + (1 - AF) * cv2.GaussianBlur(aura, (0, 0), 12))
    light = cv2.resize(cv2.GaussianBlur(light, (0, 0), 6), (OW, OH), interpolation=cv2.INTER_CUBIC)
    rlum = np.clip(EV('BL_RIM', '1.5') * rim, 0, 1.2)
    rdep = np.clip(rim * 0.95, 0, 1)
    return dict(m=m, mF=mF, rim_lum=rlum, rim_dep=rdep, light=light)


# ------------------------------------------------------------------ the big-dot sheet
def nv(*a):
    a = np.array(a, float); return a / np.linalg.norm(a)


def height(a, b, t):
    """rolling waves on the sheet (world units); a across, b up the sheet"""
    A = EV('BG_WAVE', '1.0')
    return A * (0.16 * np.sin(1.05 * a + 0.55 * b - 0.70 * t + 0.3)
                + 0.11 * np.sin(-0.62 * a + 1.25 * b - 0.52 * t + 1.9)
                + 0.07 * np.sin(1.9 * a - 0.8 * b + 0.9 * t + 4.1)
                + 0.05 * np.sin(0.35 * a + 2.3 * b - 1.1 * t + 0.7))


def sheet_point(a, b, t):
    th = np.radians(EV('BG_TILT', '38'))
    n0 = np.array([0.0, np.sin(th), -np.cos(th)])               # sheet normal, facing the camera
    h = height(a, b, t)
    X = a + h * n0[0]
    Y = EV('BG_Y0', '-1.0') + b * np.cos(th) + h * n0[1]
    Z = EV('BG_Z0', '2.4') + b * np.sin(th) + h * n0[2]
    return X, Y, Z


def project(X, Y, Z):
    f = OH * EV('BG_F', '1.15')
    return OW / 2 + f * X / Z, OH * EV('BG_CY', '0.5') - f * Y / Z


def bg_sheet(t, light):
    P = EV('BG_PITCH', '0.12')
    xs = np.arange(-7.0, 7.0 + 1e-6, P); ys = np.arange(-1.0, 9.0 + 1e-6, P)
    A_, B_ = np.meshgrid(xs, ys); A_ = A_.ravel(); B_ = B_.ravel()
    e = 1e-3
    X, Y, Z = sheet_point(A_, B_, t)
    u, v = project(X, Y, Z)
    Xa, Ya, Za = sheet_point(A_ + e, B_, t); Xb, Yb, Zb = sheet_point(A_, B_ + e, t)
    ua, va = project(Xa, Ya, Za); ub, vb = project(Xb, Yb, Zb)
    J = np.stack([np.stack([(ua - u) / e, (ub - u) / e], -1), np.stack([(va - v) / e, (vb - v) / e], -1)], -2)  # n x 2 x 2
    # surface normal (for the light)
    da = np.stack([Xa - X, Ya - Y, Za - Z], -1); db = np.stack([Xb - X, Yb - Y, Zb - Z], -1)
    nrm = np.cross(da, db); nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True)
    nrm = np.where(nrm[:, 2:3] > 0, -nrm, nrm)                          # facing the camera
    Vw = -np.stack([X, Y, Z], -1); Vw /= np.linalg.norm(Vw, axis=-1, keepdims=True)
    Lk = nv(-0.45, 0.55, -0.70)                                          # key: upper left, from the camera side
    Ls = nv(0.30 + 0.25 * np.sin(0.35 * t), 0.85, 0.55)                 # sheen: from behind / above (backlight), drifting
    lam = np.clip((nrm * Lk).sum(-1), 0, 1)
    hs = (Vw + Ls); hs /= np.linalg.norm(hs, axis=-1, keepdims=True)
    spec = np.clip((nrm * hs).sum(-1), 0, 1) ** EV('BG_SPECP', '22')
    ok = (Z > 0.3) & (u > -120) & (u < OW + 120) & (v > -120) & (v < OH + 120)
    uu, vv = np.clip(u, 0, OW - 1).astype(int), np.clip(v, 0, OH - 1).astype(int)
    lt = light[vv, uu]
    hgt = height(A_, B_, t)
    cav = np.clip(0.75 + 1.6 * hgt, 0.45, 1.25)                          # crests lift, troughs sink
    val = lt * (EV('BG_FILL', '0.18') + EV('BG_KEY', '0.55') * lam ** 1.25) * cav + EV('BG_SHEEN', '0.75') * spec * lt ** 0.6
    val = np.clip(val * EV('BG_EXP', '1.15'), 0, 1)
    # fog / distance
    val = val * np.exp(-np.maximum(Z - EV('BG_FOG0', '4.5'), 0) / EV('BG_FOG', '3.0'))
    # halftone: the disc AREA follows the tone; far rows soften (depth of field)
    rho = P * (EV('BG_R0', '0.16') + EV('BG_RK', '0.26') * np.clip(val, 0, 1) ** 0.7)
    zf = EV('BG_ZF', '3.0')
    dof = EV('BG_DOF', '5.0') * np.abs(1 / Z - 1 / zf) * zf
    img = np.zeros((OH, OW, 3), np.float32)
    idx = np.nonzero(ok & (val > 0.01))[0]
    col = L.PAL[(np.clip(0.06 + 0.94 * val, 0, 1) * 255).astype(np.uint8)]
    inten = (0.30 + 0.70 * val) * EV('BG_GAIN', '1.0')
    # sparse glints (as on the landscape): a few dots flare and fade, own phase / rate
    hr = np.random.default_rng(7); r1 = hr.random(A_.size); ph = hr.random(A_.size); kk = 0.35 + 0.5 * hr.random(A_.size)
    tw = (r1 < EV('BG_TWD', '0.025')) * np.maximum(np.sin(2 * np.pi * (kk * t + ph)), 0) ** 24 * np.clip(val * 1.5, 0, 1)
    inten = inten * (1 + 1.6 * tw)
    col = col * (1 - tw[:, None] * 0.6) + L.PAL[255][None] * tw[:, None] * 0.6
    CORE = EV('BG_CORE', '0.35')
    for i in idx:
        Jm = J[i] * rho[i]                                                # unit disc -> screen ellipse
        hx = np.hypot(Jm[0, 0], Jm[0, 1]) + dof[i] + 1.5; hy = np.hypot(Jm[1, 0], Jm[1, 1]) + dof[i] + 1.5
        x0, x1 = int(max(u[i] - hx, 0)), int(min(u[i] + hx + 1, OW)); y0, y1 = int(max(v[i] - hy, 0)), int(min(v[i] + hy + 1, OH))
        if x1 <= x0 or y1 <= y0:
            continue
        try:
            Ji = np.linalg.inv(Jm)
        except np.linalg.LinAlgError:
            continue
        gx, gy = np.meshgrid(np.arange(x0, x1) + 0.5 - u[i], np.arange(y0, y1) + 0.5 - v[i])
        sx = Ji[0, 0] * gx + Ji[0, 1] * gy; sy = Ji[1, 0] * gx + Ji[1, 1] * gy
        r = np.sqrt(sx * sx + sy * sy)                                    # 1 = the ellipse edge
        rpx = np.sqrt(abs(np.linalg.det(Jm)))                             # mean radius in px
        ew = (1.0 + dof[i]) / max(rpx, 0.5)                               # edge width (AA + defocus), in radius units
        a = np.clip((1 - r) / ew + 0.5, 0, 1)
        a = a * a * (3 - 2 * a) * (1 - CORE * np.clip(r, 0, 1) ** 2)            # a jewel: hotter core, softer rim
        k = inten[i] / (1 + 0.6 * dof[i] / max(rpx, 0.5))                  # defocus spreads, does not brighten
        img[y0:y1, x0:x1] += a[..., None] * (col[i] * k)[None, None]
    return img


def tone(img):
    Lm = L.luma(img)[..., None] if hasattr(L, 'luma') else (img @ np.array([0.114, 0.587, 0.299], np.float32))[..., None]
    sat = 1.0 + 0.30 * np.exp(-((Lm - 0.35) / 0.25) ** 2)
    img = np.clip(Lm + (img - Lm) * sat, 0, None)
    Lm = (img @ np.array([0.114, 0.587, 0.299], np.float32))[..., None]
    Lt = Lm / (1 + Lm * 0.85)
    return np.clip(img * (Lt / np.maximum(Lm, 1e-5)) * 1.25, 0, 1)


def compose(fr, t, state=None):
    F = footage(fr, state)
    bg = bg_sheet(t, F['light'])
    bloom = cv2.GaussianBlur(bg, (0, 0), 5) * EV('BG_BLOOM1', '0.35') + cv2.GaussianBlur(bg, (0, 0), 24) * EV('BG_BLOOM2', '0.30')
    bgl = bg + bloom
    mF = F['mF'][..., None]
    wrap = cv2.GaussianBlur(bgl, (0, 0), 6) * mF * cv2.GaussianBlur(1 - F['mF'], (0, 0), EV('WRAPW', '3.5'))[..., None] * EV('WRAP', '0.35')
    base = bgl * (1 - mF) + wrap
    # rim sparkles (fine dots) from the dot-sheet renderer: rim only, on black
    rim = DS.render(F['rim_dep'].astype(np.float32), 1, t, None, None, None, None, F['rim_lum'].astype(np.float32))
    # the renderer is already tone-mapped; add it in linear-ish space
    return np.clip(tone(base) + rim, 0, 1), bg


if __name__ == '__main__':
    src, fi, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    cap = cv2.VideoCapture(src)
    for _ in range(fi):
        cap.grab()
    ok, fr = cap.read()
    img, bg = compose(fr, fi / 25.0)
    cv2.imwrite(out, (img * 65535 + 0.5).astype(np.uint16))
    cv2.imwrite(out.replace('.png', '_bgonly.jpg'), (tone(bg) * 255).astype(np.uint8))
    print('done', out)
