"""2 s motion test of the depth dot-sheet look; dots ride on the footage (optical flow advection).
usage: python3 depth_sheet_clip.py depthdir start n out.mp4 src.mp4 src_offset"""
import sys, os, glob, numpy as np, cv2
import depth_sheet as DS
import mattes
from common import Writer, W, H
import common
common.FPS = 25
dd, a, n, out, src, off = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5], int(sys.argv[6])
fs = sorted(glob.glob(f'{dd}/d*.png'))[a:a + n]
# source frames (grayscale, 640x360) for flow
cap = cv2.VideoCapture(src)
for _ in range(off + a):
    cap.grab()                                   # frame-accurate (no keyframe seek)
gray = []; lumf = []
for _ in range(n):
    ok, fr = cap.read()
    lumf.append(cv2.cvtColor(cv2.resize(fr, (W, H), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY))
    gray.append(cv2.cvtColor(cv2.resize(fr, (640, 360), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY))
def matte(g8):
    g = cv2.resize(g8, (W, H)).astype(np.float32) / 255
    mt = (cv2.GaussianBlur(g, (0, 0), 2) > 0.11).astype(np.uint8)              # visible figure only (no faint trails)
    mt = cv2.morphologyEx(mt, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (45, 45)))  # keep dark bodice
    mt = cv2.dilate(mt, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))).astype(np.float32)
    return cv2.GaussianBlur(mt, (0, 0), 2.5)


MT = os.environ.get('MATTE')
smp = []
for j in range(0, n, 5):
    x = cv2.resize(cv2.imread(fs[j], -1).astype(np.float32) / 65535, (W, H))
    if MT == 'depth':
        mm, ab = mattes.depth_matte(x, gray[j]); x = ab * mm
    elif MT:
        x = x * matte(gray[j])
    smp.append(cv2.resize(x, (320, 180)))
lohi = (np.median([np.percentile(x, 2) for x in smp]), np.median([np.percentile(x, 99.7) for x in smp]))
if os.environ.get('DLOHI'):                      # fixed depth range (synthetic / flat depth)
    lohi = tuple(float(x_) for x_ in os.environ['DLOHI'].split(','))
LEVEL = None
if MT == 'depth':
    # one level per frame (her figure's 95th percentile), smoothed over time -> no pumping, no trails
    lv = []
    for j in range(n):
        x = cv2.resize(cv2.imread(fs[j], -1).astype(np.float32) / 65535, (W // 2, H // 2))
        mm, ab = mattes.depth_matte(x, cv2.resize(gray[j], (W // 2, H // 2)))
        sel = mm > 0.5
        lv.append((np.median(ab[sel]), np.percentile(ab[sel], 95)) if sel.sum() > 50 else (np.nan, np.nan))
    lv = np.array(lv)
    k = np.exp(-0.5 * (np.arange(-12, 13) / 4.0) ** 2); k /= k.sum()
    RAWLV, LEVEL = [], []
    for q in range(2):
        v = lv[:, q]; ok_ = ~np.isnan(v); v = np.interp(np.arange(n), np.nonzero(ok_)[0], v[ok_])
        RAWLV.append(v); LEVEL.append(np.convolve(np.pad(v, 12, mode='edge'), k, mode='valid'))
    RAWLV, LEVEL = np.array(RAWLV), np.array(LEVEL)          # [median, p95] per frame: measured / smoothed
    lohi = (0.0, float(np.median(LEVEL[1])))
dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
X0, Y0, LV, SHP = DS.lattice(levels=True)
if os.environ.get('DEEP'):                        # finer dot level for close push-ins (base dots keep their order + randomness)
    DS.NBASE['n'] = X0.size
    X0, Y0, LV, SHP = DS.lattice(levels='deep')
pts = (X0, Y0)
scale = np.ones(1, np.float32)            # accumulated zoom of the footage
STILL = os.environ.get('STILL')                   # write 16-bit PNG stills (any output scale) instead of a video
wr = None if STILL else Writer(out, None, [])
TABS = os.environ.get('T0ABS')                    # time measured from the start of the clip (matches the full render)
ST = {} if os.environ.get('DOTSMOOTH', '1') == '1' else None
USELUM = os.environ.get('USELUM')
LREF = np.percentile(np.stack(lumf[::5]), 99.5) if USELUM else 1.0
NOTRACK = os.environ.get('NOTRACK')
if os.environ.get('ORBIT'):                      # specks orbit her: centre = her (smoothed) lit centroid
    cxs, cys = [], []
    for j in range(n):
        ys_, xs_ = np.nonzero(lumf[j] > 25)
        cxs.append(xs_.mean() if len(xs_) else W / 2); cys.append(ys_.mean() if len(ys_) else H / 2)
    kk = np.exp(-0.5 * (np.arange(-15, 16) / 6.0) ** 2); kk /= kk.sum()
    CXS = np.convolve(np.pad(cxs, 15, mode='edge'), kk, mode='valid'); CYS = np.convolve(np.pad(cys, 15, mode='edge'), kk, mode='valid')
HC = {}
CAMMAG = [1.0]
SWAY = float(os.environ.get('SWAYSPK', '0'))
if SWAY:
    # specks follow the subject's sway: mean flow on the lit subject, integrated -> position, eased with a lag
    # (they drift after it, like dust in the air it moves through)
    dis0 = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
    vel = [np.zeros(2)]
    for j in range(n - 1):
        fl0 = dis0.calc(gray[j], gray[j + 1], None); m0 = gray[j] > 40
        vel.append(np.array([fl0[..., 0][m0].mean(), fl0[..., 1][m0].mean()]) * (W / 640.0) if m0.sum() > 200 else np.zeros(2))
    pos = np.cumsum(np.array(vel), 0)
    pos -= np.linspace(pos[0], pos[-1], n)                            # remove slow drift: sway only
    lagp = np.zeros_like(pos); acc = pos[0].copy(); vv = np.zeros(2)
    for j in range(n):                                                # critically-damped follow (~0.35 s): drag + catch-up
        wn = 2 * np.pi / 1.4; dt = 1 / 25.0
        vv += (wn * wn * (pos[j] - acc) - 2 * wn * vv) * dt; acc = acc + vv * dt; lagp[j] = acc
    SWP = lagp * SWAY
    print('sway amplitude px', np.abs(pos).max(0).round(1), 'speck offset max', np.abs(SWP).max(0).round(1))
RNG_ = [int(x_) for x_ in os.environ['RANGE'].split(',')] if os.environ.get('RANGE') else None   # render a slice (warm-up frames before it)
for i, f in enumerate(fs):
    if RNG_ and (i < RNG_[0] - 6 or i >= RNG_[1]):
        continue
    if SWAY:
        DS.SPKOFF[:] = SWP[i]
    dep = DS.load_depth(f, lohi)
    if MT:                                       # subject on true black: keep only the figure
        dep = cv2.resize(cv2.imread(f, -1).astype(np.float32) / 65535, (W, H))
        if MT == 'depth':
            mm, ab = mattes.depth_matte(dep, gray[i])
            m0, q0 = RAWLV[:, i]; m1, q1 = LEVEL[:, i]                 # remap this frame's level/spread to the smoothed ones
            ab = np.clip((ab - m0) * (q1 - m1) / max(q0 - m0, 1e-3) + m1, 0, 1)
            if os.environ.get('THIN'):
                # slim cut-out: the depth matte only where the footage confirms the figure (lit, or right next to lit)
                lit = (cv2.GaussianBlur(lumf[i].astype(np.float32) / 255, (0, 0), 1.5) > float(os.environ.get('THIN_T', '0.06'))).astype(np.uint8)
                lit = cv2.morphologyEx(lit, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)))
                lit = cv2.GaussianBlur(cv2.dilate(lit, np.ones((3, 3), np.uint8)).astype(np.float32), (0, 0), 1.2)
                if os.environ.get('EDGEMATTE'):
                    # edge from the full-res footage (smooth, anti-aliased); depth only says WHAT is subject:
                    # its coarse outline is pushed outside the light edge, deep inside it holds dark shadows
                    lf_ = cv2.GaussianBlur(lumf[i].astype(np.float32) / 255, (0, 0), 1.0)
                    e0, e1 = (float(x_) for x_ in os.environ.get('EDGEMATTE_T', '0.035,0.11').split(','))
                    la = np.clip((lf_ - e0) / (e1 - e0), 0, 1); la = la * la * (3 - 2 * la)
                    outer = cv2.GaussianBlur(cv2.dilate(mm, np.ones((15, 15), np.uint8)), (0, 0), 3)
                    inner = cv2.GaussianBlur(cv2.erode(mm, np.ones((13, 13), np.uint8)), (0, 0), 4)
                    mm = np.maximum(la, inner) * np.clip(outer, 0, 1)
                    # presence ramps over ~one dot spacing at the edge -> partial dots, no row-by-row staircase
                    DS.PRESMAP['a'] = np.clip((cv2.GaussianBlur(mm, (0, 0), DS.PITCH_PX * 0.6) - 0.45) / 0.45, 0, 1).astype(np.float32)   # ramps INSIDE the edge
                else:
                    mm = cv2.erode(mm, np.ones((5, 5), np.uint8)) * lit
            dep = ab * mm
        else:
            dep = dep * matte(gray[i])
        dep = np.clip((dep - lohi[0]) / (lohi[1] - lohi[0] + 1e-6), 0, 1)
    lm = (lumf[i].astype(np.float32) / LREF) if USELUM else None
    if os.environ.get('ORBIT'):
        DS.ORBIT['cx'] = float(CXS[i]); DS.ORBIT['cy'] = float(CYS[i]) * 0.5 + H * 0.25
        if os.environ.get('FIT'):                  # orbit / swirl axis in the fitted screen frame too
            fx, fy = DS.fit_frame(np.array([DS.ORBIT['cx']]), np.array([DS.ORBIT['cy']]))
            DS.ORBIT['cx'], DS.ORBIT['cy'] = float(fx[0]), float(fy[0])
    if os.environ.get('TRANS'):                    # forming / dissolving + speck fades, timed to 146 BPM
        BT = 60.0 / 146
        TL = float(os.environ.get('CLIPLEN', n / 25.0))
        isp, idu = (float(x_) for x_ in os.environ.get('TRANS_IN', '1.6,1.7').split(','))
        if os.environ.get('TRANS_MODE') == 'hands':
            osp, odu = (float(x_) for x_ in os.environ.get('TRANS_OUT', '2.2,2.6').split(','))
            fdx, fdy = (float(x_) for x_ in os.environ.get('FLOW_DIR', '-1,0').split(','))
            if 'ocx' not in HC:                        # dispersal centre: the subject's lit centroid over the last second
                xs_, ys_ = [], []
                for j in range(max(0, n - 25), n):
                    yy_, xx_ = np.nonzero(lumf[j] > 25)
                    if len(xx_):
                        xs_.append(xx_.mean()); ys_.append(yy_.mean())
                HC['ocx'] = float(np.mean(xs_)) if xs_ else W / 2; HC['ocy'] = float(np.mean(ys_)) if ys_ else H / 2
            ocx, ocy = np.array([HC['ocx']]), np.array([HC['ocy']])
            if os.environ.get('FIT'):
                ocx, ocy = DS.fit_frame(ocx, ocy)
            _dz = DS.push_dz((a + i) / 25.0 if TABS else i / 25.0)
            ocx, ocy = DS.push_xy(ocx, ocy, DS.Z0 - DS.REL * 0.8, _dz)
            DS.TRANS.update(dict(mode='hands', fly_soft=float(os.environ.get('FLY_SOFT', '0.8')), fdir=(fdx, fdy), in_xmin=float(os.environ.get('IN_XMIN', '0.45')),
                                 in0=0.0, in_spread=isp * BT, in_dur=idu * BT,
                                 out0=TL - (osp * BT + float(os.environ.get('OUTK', '1.25')) * odu * BT) - float(os.environ.get('OUTPAD', '0.15')), out_spread=osp * BT, out_dur=odu * BT,
                                 ocx=float(np.ravel(ocx)[0]), ocy=float(np.ravel(ocy)[0])))
        else:
          DS.TRANS.update(dict(dir=float(os.environ.get('TRANS_DIR', '1')), os=float(os.environ.get('TRANS_OS', '0.8')),
                             in0=0.0, in_spread=isp * BT, in_dur=idu * BT,
                             out0=(1e9 if os.environ.get('TRANS_NOOUT') else TL - 3.2 * BT), out_spread=1.2 * BT, out_dur=1.5 * BT,
                             sp_in0=0.0, sp_in_spread=4 * BT, sp_fade=1.0 * BT,
                             sp_out0=TL - 4.2 * BT, sp_out_spread=3.0 * BT))
    tt_ = (a + i) / 25.0 if TABS else i / 25.0
    def campath2(tq):
        # one continuous move: the camera starts drifting while the last particles land and only ever accelerates
        # (no settle-then-go hand-over); the dive is the same move speeding up, and the bouquet peels apart around it
        ss = lambda x: (lambda c: c * c * c * (c * (c * 6 - 15) + 10))(np.clip(x, 0, 1))
        TLc = float(os.environ.get('CLIPLEN', n / 25.0)); deg = np.pi / 180
        Zs = DS.Z0 - DS.REL * 0.8
        if 'cp' not in HC:
            tg = np.linspace(0, TLc, 4001); dt = tg[1] - tg[0]
            c0 = float(os.environ.get('CAM0', '1.5')); te = TLc - 0.2
            va = 0.8 * ss((tg - c0) / 1.2) + np.clip((tg - (c0 + 0.5)) / (te - c0 - 0.5), 0, None) ** 1.0      # drift -> momentum
            d0 = float(os.environ.get('DIVE0', '7.4'))
            vb = np.clip((tg - d0) / (te - d0), 0, None) ** 2.0                                               # the dive
            Ia = np.cumsum(va) * dt; Ib = np.cumsum(vb) * dt
            tm = float(os.environ.get('MAGT', '8.5')); mg = float(os.environ.get('MAGV', '1.7'))
            A_ = np.array([[np.interp(tm, tg, Ia), np.interp(tm, tg, Ib)], [np.interp(te, tg, Ia), np.interp(te, tg, Ib)]])
            ab = np.linalg.solve(A_, [Zs * (1 - 1 / mg), Zs + float(os.environ.get('FLY_OVER', '0.15'))])
            print('campath: drift/momentum %.4f dive %.4f' % tuple(ab), flush=True)
            HC['cp'] = (tg, ab[0] * Ia + ab[1] * Ib, Ia / np.interp(te, tg, Ia))
        tg, DOL, ORB = HC['cp']
        dol = float(np.interp(tq, tg, DOL)); ob = float(np.interp(tq, tg, ORB))
        s0 = float(os.environ.get('BURST0', '7.4')); bsp = float(os.environ.get('BURST_SP', '1.2')); bdu = float(os.environ.get('BURST_DU', '0.85'))
        DS.CAM.update(dyaw=float(os.environ.get('ORBIT_DIR', '1')) * float(os.environ.get('ORBIT_DEG', '33')) * ob * deg,
                      dtilt=4 * ob * deg, dolly=dol,
                      explode=ss((tq - 3.0) / 4.0) * float(os.environ.get('EXPLODE', '1.0')), gap=float(os.environ.get('GAP', '0.09')),
                      burst=1.0 if tq > s0 - 0.05 else 0.0, stag=(s0, bsp, bdu), bt=tq,
                      focus=dol * float(os.environ.get('FOCUSK', '1.0')))
        if os.environ.get('AIM'):                  # footage px of the rose heart the camera flies into
            gxa, gya = (float(x_) for x_ in os.environ['AIM'].split(','))
            DS.CAM['aim'] = ((gxa - W / 2) / H, -(gya - H / 2) / H, Zs, float(ss((tq - float(os.environ.get('AIM0', '2.5'))) / float(os.environ.get('AIM_DU', '6.0')))))
        CAMMAG[0] = float(min(Zs / max(Zs - dol, 0.05), 2.8 if os.environ.get('DEEP') else 2.2))
        if 'g0' not in HC:
            HC['g0'] = float(os.environ.get('GAIN', '1'))
        os.environ['GAIN'] = str(HC['g0'] * (1 + float(os.environ.get('PUSHGAIN', '0')) * ob))   # crisp dots carry less light than the old soft blobs
    if os.environ.get('CAMPATH') == 'rose2':
        campath2(tt_)
        if SWAY:
            DS.SPKOFF[0] += -DS.CAM['dyaw'] * 900
    if os.environ.get('CAMPATH') == 'rose':
        # settle (2.1-4.0 s): slow 5 deg orbit + 10 % push; momentum (from MOM0): orbit to ~30 deg, push to ~50 %,
        # petals separate into layers; the camera keeps travelling through the dissolve
        ss = lambda x: (lambda c: c * c * c * (c * (c * 6 - 15) + 10))(np.clip(x, 0, 1))
        m0 = float(os.environ.get('MOM0', '4.0')); TLc = float(os.environ.get('CLIPLEN', n / 25.0))
        s1 = ss((tt_ - 2.1) / (m0 - 2.1 + 0.4)); s2 = ss((tt_ - (m0 - 0.4)) / (TLc - m0 + 0.4))
        deg = np.pi / 180
        DS.CAM['dyaw'] = float(os.environ.get('ORBIT_DIR', '1')) * (5 * s1 + 25 * s2) * deg
        DS.CAM['dtilt'] = 4 * s2 * deg
        Zs = DS.Z0 - DS.REL * 0.8
        mag = 1 + 0.10 * s1 + float(os.environ.get('PUSHMAX', '0.40')) * s2
        DS.CAM['dolly'] = Zs * (1 - 1 / mag); CAMMAG[0] = mag
        DS.CAM['explode'] = ss((tt_ - (m0 + 0.3)) / 3.9) * float(os.environ.get('EXPLODE', '1.0'))
        DS.CAM['gap'] = float(os.environ.get('GAP', '0.09'))
        if SWAY:                                     # specks drift with the orbit too (parallax)
            DS.SPKOFF[0] += -DS.CAM['dyaw'] * 900
    if NOTRACK and os.environ.get('CAMPATH') == 'rose2' and DS.CAM.get('burst', 0) > 0:
        acc_ = None; KS = 4
        for ks in range(KS):                         # open shutter: 4 sub-frames over half a frame
            tq_ = tt_ + (ks - (KS - 1) / 2) / KS * 0.5 / 25
            campath2(tq_)
            sub_ = DS.render(dep, 1, tq_, (X0, Y0), DS.level_vis(LV, np.full(X0.size, CAMMAG[0], np.float32)), None, ST if ks == KS - 1 else None, lm)
            acc_ = sub_ if acc_ is None else acc_ + sub_
        fr_ = acc_ / KS
    elif NOTRACK and os.environ.get('CAMPATH'):
        # pushing in: in-between dots fade in with the magnification -> constant dot density, stable identities
        fr_ = DS.render(dep, 1, tt_, (X0, Y0), DS.level_vis(LV, np.full(X0.size, CAMMAG[0], np.float32)), None, ST, lm)
    elif NOTRACK:
        fr_ = DS.render(dep, 1, tt_, None, None, None, ST, lm)
    else:
        fr_ = DS.render(dep, 1, tt_, pts, DS.level_vis(LV, DS.local_scale(pts, SHP)), None, ST, lm)
    if RNG_ and i < RNG_[0]:
        continue
    if STILL:
        cv2.imwrite(f'{STILL}_{a + i:04d}.png', (np.clip(fr_, 0, 1) * 65535 + 0.5).astype(np.uint16))
    else:
        wr.write(fr_)
    if i + 1 < n and not NOTRACK:
        fl = dis.calc(gray[i], gray[i + 1], None)
        fl = cv2.resize(fl, (W, H)) * (W / 640.0)
        # 1) one coherent motion for the whole flower: homography fitted (RANSAC) to the flow on the flower
        calm = np.clip((cv2.GaussianBlur(dep, (0, 0), 8) - 0.12) / 0.25, 0, 1)
        yy, xx = np.mgrid[8:H:16, 8:W:16]
        m = calm[yy, xx] > 0.5
        p0 = np.stack([xx[m], yy[m]], -1).astype(np.float32)
        p1 = p0 + fl[yy[m], xx[m]]
        Hm = None
        if len(p0) > 40 and os.environ.get('SIMILARITY'):
            A, _ = cv2.estimateAffinePartial2D(p0, p1, method=cv2.RANSAC, ransacReprojThreshold=1.5)
            if A is not None:
                sc = np.hypot(A[0, 0], A[1, 0]); k = np.clip(sc, 0.996, 1.004) / sc
                A[:, :2] *= k
                Hm = np.vstack([A, [0, 0, 1]])
        elif len(p0) > 40:
            Hm, _ = cv2.findHomography(p0, p1, cv2.RANSAC, 1.5)
        if Hm is None:
            Hm = np.eye(3)
        scale = scale * np.sqrt(abs(np.linalg.det(Hm[:2, :2] / Hm[2, 2])))
        P = np.stack([pts[0], pts[1], np.ones_like(pts[0])], 0).astype(np.float64)
        Q = Hm @ P; qx, qy = (Q[0] / Q[2]).astype(np.float32), (Q[1] / Q[2]).astype(np.float32)
        # 2) gentle local correction (opening petals): heavily smoothed residual flow, weighted to the flower
        gx_, gy_ = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
        G = np.stack([gx_, gy_, np.ones_like(gx_)], 0).reshape(3, -1)
        GQ = (Hm @ G); hx = (GQ[0] / GQ[2]).reshape(H, W) - gx_; hy = (GQ[1] / GQ[2]).reshape(H, W) - gy_
        res = (fl - np.stack([hx, hy], -1)) * calm[..., None]
        res = cv2.GaussianBlur(res, (0, 0), 28) / (cv2.GaussianBlur(calm, (0, 0), 28)[..., None] + 1e-3) * calm[..., None]
        res = np.clip(res, -3, 3) * 0.8
        s = DS.sampler(pts)
        pts = (qx + s(res[..., 0]), qy + s(res[..., 1]))
if wr: wr.close()
if not os.environ.get("DEEP"): ls_ = DS.local_scale(pts, SHP); print('final zoom global %.2f, local p50 %.2f p95 %.2f' % (scale[0], np.median(ls_), np.percentile(ls_, 95)))
