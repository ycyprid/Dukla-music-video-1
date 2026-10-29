"""Night-sky shot in the halftone look: every star is its own glistening particle, the Milky Way glow and the trees
are the dot sheet.
  1. sky motion: consecutive star matches -> similarity transforms, chained and smoothed over time (the footage has
     duplicate frames; the smooth fit removes their hitch) and resampled to 25 fps
  2. a clean sky mosaic (all frames aligned + averaged, trees masked out) -> star catalogue (sub-pixel, flux) + glow
  3. per output frame: glow warped into place + the footage's own trees -> sheet luminance; stars -> EXTRA particles
     (own breathing + own sparse flashes, cross glints on the brightest, hidden behind the trees)
usage: python3 stars_clip.py src.mp4 out.mp4"""
import sys, os
import numpy as np, cv2
from scipy.spatial import cKDTree
import common
common.FPS = 25
import depth_sheet as DS
from common import W, H, Writer

src, out = sys.argv[1], sys.argv[2]
cap = cv2.VideoCapture(src); SFPS = cap.get(cv2.CAP_PROP_FPS) or 24.0
G = []
while True:
    ok, f = cap.read()
    if not ok:
        break
    G.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))                  # uint8 (memory)
N = len(G); FH, FW = G[0].shape[:2]
print('frames', N, 'fps', SFPS)


def stars(g, th):
    hp = g - cv2.GaussianBlur(g, (0, 0), 3)
    mx = cv2.dilate(hp, np.ones((9, 9), np.uint8))
    ys, xs = np.nonzero((hp == mx) & (hp > th))
    return np.stack([xs, ys], -1).astype(np.float32), hp


# ---- static tree mask (trees don't move; the sky drifts behind them) ----
med = np.median(np.stack(G[0:120:4]), 0).astype(np.float32)                       # stars move -> median = glow + dark trees
medb = cv2.GaussianBlur(med, (0, 0), 3)
skylev = cv2.GaussianBlur(cv2.dilate(cv2.resize(medb, (192, 108), interpolation=cv2.INTER_AREA), np.ones((9, 9), np.uint8)), (0, 0), 6)
skylev = cv2.resize(skylev, (FW, FH), interpolation=cv2.INTER_CUBIC)
tree = np.clip((0.62 * skylev - medb) / (0.18 * skylev + 1e-3), 0, 1)
tree = cv2.morphologyEx(tree, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
tree = cv2.GaussianBlur(tree, (0, 0), 2.0)
print('tree fraction %.3f' % (tree > 0.5).mean())

# ---- sky motion ----
P = [stars(g.astype(np.float32) * (1 - tree), 70)[0] for g in G]
Ms = [np.eye(3)]
for j in range(N - 1):
    d, ix = cKDTree(P[j + 1]).query(P[j], distance_upper_bound=8)
    ok_ = np.isfinite(d)
    Hm = None
    if ok_.sum() > 30:                                            # sky rotating past a still camera = a homography
        Hm, _ = cv2.findHomography(P[j][ok_], P[j + 1][ix[ok_]], cv2.RANSAC, 1.0)
    if Hm is None:
        Hm = np.eye(3)
    M_ = Hm @ Ms[-1]; Ms.append(M_ / M_[2, 2])                    # frame-0 coords -> frame-j coords
# smooth fit of the 8 homography parameters vs time (duplicate frames -> zero steps, smoothed away)
prm = np.array([M.ravel()[:8] for M in Ms])
tj = np.arange(N) / SFPS
coef = [np.polyfit(tj, prm[:, k], 5) for k in range(8)]


def Mt(t):                                                        # smoothed frame-0 -> frame(t) transform
    return np.append([np.polyval(c, t) for c in coef], 1.0).reshape(3, 3)


chk = np.array([[200, 200, 1], [1700, 200, 1], [200, 900, 1], [1700, 900, 1], [960, 540, 1.0]]).T
err = []
for j in range(N):
    a_ = Ms[j] @ chk; b_ = Mt(tj[j]) @ chk
    err.append(np.abs(a_[:2] / a_[2] - b_[:2] / b_[2]).max())
print('motion fit residual px: max %.2f median %.2f' % (np.max(err), np.median(err)))


# ---- clean sky mosaic in frame-0 coords (canvas with margin where the sky came from) ----
c0 = np.concatenate([(lambda q: q / q[2])(np.linalg.inv(Ms[j]) @ np.array([[0, FW, 0, FW], [0, 0, FH, FH], [1, 1, 1, 1.0]]))
                     for j in range(0, N, 10)] + [np.array([[0, FW, 0, FW], [0, 0, FH, FH], [1, 1, 1, 1.0]])], 1)
OX = int(np.ceil(max(0, -c0[0].min()))) + 40; OY = int(np.ceil(max(0, -c0[1].min()))) + 40
CW = int(np.ceil(max(FW, c0[0].max()))) + OX + 40; CH = int(np.ceil(max(FH, c0[1].max()))) + OY + 40
Toff = np.array([[1, 0, OX], [0, 1, OY], [0, 0, 1.0]])
acc = np.zeros((CH, CW), np.float32); cnt = np.zeros((CH, CW), np.float32)
skyw = (1 - tree).astype(np.float32)
edge = np.ones((FH, FW), np.float32); edge[:6] = edge[-6:] = 0; edge[:, :6] = edge[:, -6:] = 0
for j in range(N):
    Minv = Toff @ np.linalg.inv(Ms[j])                             # frame j -> canvas (actual registration)
    acc += cv2.warpPerspective(G[j].astype(np.float32) * skyw * edge, Minv, (CW, CH), flags=cv2.INTER_LINEAR)
    cnt += cv2.warpPerspective(skyw * edge, Minv, (CW, CH), flags=cv2.INTER_LINEAR)
mos = acc / np.maximum(cnt, 1e-3); valid = cnt > 2.0
fillv = np.median(mos[valid]); mos[~valid] = fillv
print('mosaic', CW, CH, 'valid %.2f' % valid.mean())
# glow (stars removed) and the star catalogue
glow = cv2.GaussianBlur(cv2.medianBlur(np.clip(mos, 0, 255).astype(np.uint8), 7).astype(np.float32), (0, 0), 3)
hp = mos - cv2.GaussianBlur(mos, (0, 0), 2.5)
mx = cv2.dilate(hp, np.ones((7, 7), np.uint8))
# ---- star tracks: every star followed through the footage itself (lens distortion / drift can't smear it) ----
g2 = G[N // 2].astype(np.float32); h0 = g2 - cv2.GaussianBlur(g2, (0, 0), 2.5)
sig0 = 1.4826 * np.median(np.abs(h0[tree < 0.1] - np.median(h0[tree < 0.1])))
THT = float(os.environ.get('STAR_SIG', '7')) * sig0
print('per-frame hp noise %.1f, star threshold %.1f' % (sig0, THT))


def detect(g):
    gf = g.astype(np.float32); hp_ = gf - cv2.GaussianBlur(gf, (0, 0), 2.5)
    mx_ = cv2.dilate(hp_, np.ones((7, 7), np.uint8))
    ys_, xs_ = np.nonzero((hp_ == mx_) & (hp_ > THT) & (tree < 0.5) & (edge > 0))
    sx_ = np.zeros(len(xs_)); sy_ = np.zeros(len(xs_)); sw_ = np.zeros(len(xs_))
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            w_ = np.clip(hp_[np.clip(ys_ + dy, 0, FH - 1), np.clip(xs_ + dx, 0, FW - 1)], 0, None)
            sx_ += w_ * dx; sy_ += w_ * dy; sw_ += w_
    return np.stack([xs_ + sx_ / np.maximum(sw_, 1e-3), ys_ + sy_ / np.maximum(sw_, 1e-3)], -1), hp_[ys_, xs_]


D = [detect(g) for g in G]
print('stars per frame ~', int(np.median([len(d[1]) for d in D])))


def qbasis(p):
    x = p[:, 0] / FW - 0.5; y = p[:, 1] / FH - 0.5
    return np.stack([np.ones_like(x), x, y, x * x, x * y, y * y], -1)


tid = [np.full(len(D[j][1]), -1, np.int64) for j in range(N)]
step = np.zeros(N)                                                # sky progress per frame step (px)
nxt = 0
for j in range(N - 1):
    p, q = D[j][0], D[j + 1][0]
    if len(p) < 10 or len(q) < 10:
        continue
    d, ix = cKDTree(q).query(p, distance_upper_bound=8)
    ok_ = np.isfinite(d)
    dv = q[ix[ok_]] - p[ok_]; med_ = np.median(dv, 0)
    good = np.linalg.norm(dv - med_, axis=1) < 2.5
    B = qbasis(p[ok_][good])
    cx_ = np.linalg.lstsq(B, dv[good, 0], rcond=None)[0]; cy_ = np.linalg.lstsq(B, dv[good, 1], rcond=None)[0]
    Bp = qbasis(p); pred = p + np.stack([Bp @ cx_, Bp @ cy_], -1)   # local motion field -> predicted positions
    step[j + 1] = np.median(np.hypot(Bp @ cx_, Bp @ cy_))
    tq = cKDTree(q); d1, i1 = tq.query(pred, distance_upper_bound=2.0)
    d2, i2 = cKDTree(pred).query(q, distance_upper_bound=2.0)
    for a_ in np.nonzero(np.isfinite(d1))[0]:
        b_ = i1[a_]
        if i2[b_] != a_:
            continue                                              # mutual nearest only
        if tid[j][a_] < 0:
            tid[j][a_] = nxt; nxt += 1
        tid[j + 1][b_] = tid[j][a_]
sky = np.cumsum(step)                                             # measured sky progress (duplicates -> flat)
fj = np.arange(N, dtype=float)
sc_ = np.polyfit(fj, sky, 3)                                      # smooth progress: no hitch at duplicate frames
# gather tracks
rows = []
for j in range(N):
    m_ = tid[j] >= 0
    rows.append(np.column_stack([tid[j][m_], np.full(m_.sum(), j), D[j][0][m_], D[j][1][m_]]))
R_ = np.concatenate(rows); R_ = R_[np.argsort(R_[:, 0], kind='stable')]
ids_, st_, ln_ = np.unique(R_[:, 0], return_index=True, return_counts=True)
MINL = int(os.environ.get('STAR_MINLEN', '10'))
TX, TY, S0, S1, FL = [], [], [], [], []
for k in np.nonzero(ln_ >= MINL)[0]:
    rr_ = R_[st_[k]:st_[k] + ln_[k]]
    sv_ = sky[rr_[:, 1].astype(int)]
    if np.ptp(sv_) < 1e-3:
        continue
    deg = 1 if ln_[k] < 25 else (2 if ln_[k] < 80 else 3)
    cxk = np.polyfit(sv_, rr_[:, 2], deg); cyk = np.polyfit(sv_, rr_[:, 3], deg)
    TX.append(np.pad(cxk, (3 - deg, 0))); TY.append(np.pad(cyk, (3 - deg, 0)))
    S0.append(sv_.min()); S1.append(sv_.max()); FL.append(np.median(rr_[:, 4]))
TX, TY, S0, S1, fl = np.array(TX), np.array(TY), np.array(S0), np.array(S1), np.array(FL)
print('star tracks', len(fl), 'median length %.0f frames' % np.median((S1 - S0) / max(np.mean(step[1:]), 1e-3)))
KEEP = float(os.environ.get('STAR_KEEP', '0.45'))                  # the brighter part of the sky only: a clear hierarchy
kk_ = fl >= np.quantile(fl, 1 - KEEP)
TX, TY, S0, S1, fl = TX[kk_], TY[kk_], S0[kk_], S1[kk_], fl[kk_]
NS = len(fl)
rank = np.argsort(np.argsort(fl)) / max(NS - 1, 1)
fn = rank                                                         # 0 faint .. 1 brightest (by rank)
SPD = max(np.mean(step[1:]), 1e-3)                                # px per source frame
rng = np.random.default_rng(2026)
f1 = rng.uniform(0.10, 0.35, NS); p1 = rng.random(NS)                              # slow breathing
f2 = rng.uniform(0.04, 0.18, NS); p2 = rng.random(NS)                             # sparse sharp flashes (own rhythm)
f3 = rng.uniform(0.9, 2.2, NS); p3 = rng.random(NS)                               # fast shimmer during a flash
flashy = rng.random(NS) < (0.12 + 0.55 * fn)
cross = fn > np.percentile(fn, float(os.environ.get('CROSS_P', '98.8')))
tin = rng.uniform(0.0, 0.9, NS) * (1.0 - 0.4 * fn); tout = rng.uniform(0.0, 0.6, NS) * (1.0 - 0.4 * fn)
ci = np.where(fn > 0.55, rng.uniform(238, 255, NS), np.where(fn > 0.2, rng.uniform(205, 240, NS), rng.uniform(150, 215, NS)))
COL = DS.L.PAL[ci.astype(np.uint8)]
TL = float(os.environ.get('CLIPLEN', N / SFPS))
NO = int(round(TL * 25))
S = W / FW                                                         # footage px -> design px
tree_d = cv2.resize(tree, (W, H), interpolation=cv2.INTER_AREA)
GLOWG = float(os.environ.get('GLOWG', '1.0'))
# sheet luminance reference: glow + lit trees over the clip
LREF = float(os.environ.get('LREF', '0'))
NEB = os.environ.get('NEBULA')
if NEB:
    # Milky Way as fine stardust: particles sampled from the glow (in sky coords, so they travel with the sky)
    gv = glow[valid]; fl0 = np.percentile(gv, float(os.environ.get('NEB_FLOOR_P', '45'))); fl1 = np.percentile(gv, 99.7)
    dens = np.clip((glow - fl0) / (fl1 - fl0), 0, 1) ** float(os.environ.get('NEB_G', '1.6')) * valid
    pr = dens.ravel() / dens.sum()
    ND = int(os.environ.get('NEB_N', '40000'))
    rgn = np.random.default_rng(11)
    pick = rgn.choice(dens.size, ND, p=pr)
    DY, DX = np.divmod(pick, CW)
    DX = DX + rgn.random(ND) - OX; DY = DY + rgn.random(ND) - OY        # frame-0 coords
    dd_ = dens.ravel()[pick]
    DW_ = (0.25 + 0.75 * dd_) * rgn.uniform(0.3, 1.0, ND) ** 2 * float(os.environ.get('NEB_K', '0.05'))
    DCi = np.clip(110 + 140 * dd_ ** 0.9 + rgn.normal(0, 14, ND), 95, 250)
    DCOL = DS.L.PAL[DCi.astype(np.uint8)]
    DPH = rgn.random(ND); DFQ = rgn.uniform(0.08, 0.3, ND)
    # tree rim: particles along the silhouette edge, lit by the sky behind it
    tb = (tree_d > 0.5).astype(np.uint8)
    nl_, lb_, st2, _ = cv2.connectedComponentsWithStats(1 - tb, 4)          # fill small sky holes inside the foliage
    for k in range(1, nl_):
        if st2[k, cv2.CC_STAT_AREA] < 4000:
            tb[lb_ == k] = 1
    edge_ = cv2.morphologyEx(tb, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) & (1 - cv2.erode(tb, np.ones((3, 3), np.uint8)))
    ey_, ex_ = np.nonzero(edge_)
    ex_ = ex_ + rgn.uniform(-0.4, 0.4, ex_.size); ey_ = ey_ + rgn.uniform(-0.4, 0.4, ey_.size)
    RPH = rgn.random(ex_.size); RW_ = rgn.uniform(0.4, 1.0, ex_.size)
    print('nebula particles', ND, 'rim particles', ex_.size)
wr = Writer(out, None, [])
ST = {}
sm = lambda x: x * x * (3 - 2 * x)
for i in range(int(os.environ.get('START', '0')), NO):
    t = i / 25.0
    M = Mt(t)
    # sheet: the glow warped into place + the footage's own trees (their light changes)
    gm = cv2.warpPerspective(glow, M @ np.linalg.inv(Toff), (FW, FH), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    js = min(int(round(t * SFPS)), N - 1)
    tl = cv2.GaussianBlur(G[js].astype(np.float32), (0, 0), 1.2)
    lumf = gm * (1 - tree) + tl * tree
    lumd = cv2.resize(lumf, (W, H), interpolation=cv2.INTER_AREA) / 255.0
    if LREF <= 0:                                                 # fixed per clip: sky base -> black, Milky Way core -> GLOW_TOP
        skyv = lumd[tree_d < 0.2]
        FLOOR = np.percentile(skyv, float(os.environ.get('GLOW_FLOOR_P', '35')))
        LREF = (np.percentile(skyv, 99.5) - FLOOR) / float(os.environ.get('GLOW_TOP', '0.5'))
        print('glow floor %.3f lref %.3f' % (FLOOR, LREF))
    lum = np.clip((lumd - FLOOR) / LREF, 0, 1.3) ** GLOWG
    dep = (0.55 + 0.25 * tree_d).astype(np.float32)
    # stars: smoothed sky motion, hidden behind the trees, own glisten
    sv = np.polyval(sc_, t * SFPS)                                # smooth sky progress at this moment
    svc = np.clip(sv, S0 - 8 * SPD, S1 + 8 * SPD)                 # tiny extrapolation past each track's ends
    u = ((TX[:, 0] * svc + TX[:, 1]) * svc + TX[:, 2]) * svc + TX[:, 3]
    v = ((TY[:, 0] * svc + TY[:, 1]) * svc + TY[:, 2]) * svc + TY[:, 3]
    u = u * S; v = v * S
    fs_ = 7 * SPD                                                 # track ends: soft (unless at the clip's ends)
    ea = np.where(S0 <= sky[1] + 1e-6, 1.0, np.clip((sv - S0 + 4 * SPD) / fs_, 0, 1))
    eb = np.where(S1 >= sky[-2] - 1e-6, 1.0, np.clip((S1 + 4 * SPD - sv) / fs_, 0, 1))
    inside = (u > -20) & (u < W + 20) & (v > -20) & (v < H + 20) & (sv > S0 - 8 * SPD) & (sv < S1 + 8 * SPD)
    occ = cv2.remap(tree_d, u.astype(np.float32).reshape(1, -1), v.astype(np.float32).reshape(1, -1), cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)[0]
    vis = np.clip(1 - occ * 1.3, 0, 1) * inside
    a_ = np.clip((t - tin) / 0.6, 0, 1); b_ = np.clip((TL - 0.15 - t - tout) / 0.6, 0, 1)
    vis = vis * sm(a_) * sm(b_)                                    # each star glistens on / off on its own
    vis = vis * sm(ea) * sm(eb)
    breathe = 0.62 + 0.38 * np.sin(np.pi * (f1 * t + p1)) ** 2
    fl_ = np.maximum(np.sin(2 * np.pi * (f2 * t + p2)), 0) ** 40 * flashy
    shim = 0.7 + 0.3 * np.sin(2 * np.pi * (f3 * t + p3))
    base = (0.015 + 2.8 * fn ** 5.5)
    w = base * breathe * (1 + 3.2 * fl_ * shim * (0.4 + 0.6 * fn)) * vis * float(os.environ.get('STAR_K', '1.0'))
    U, V, C, Wt = [u], [v], [COL], [w]
    # cross glints on the brightest while they flash (4 fading arms)
    HERO = float(os.environ.get('HERO', '0'))
    cm = cross & (((fl_ > 0.05) | (HERO > 0))) & (vis > 0.05)
    if cm.any():
        k_ = np.nonzero(cm)[0]
        Lr = (6 + 16 * fn[k_]) * np.maximum(fl_[k_], HERO)
        for (ax, ay) in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            for s_ in np.linspace(0.15, 1.0, 7):
                U.append(u[k_] + ax * s_ * Lr); V.append(v[k_] + ay * s_ * Lr)
                C.append(DS.L.PAL[np.full(k_.size, 252)]); Wt.append(w[k_] * 0.16 * (1 - s_) ** 1.6)
    if NEB:
        q = (M @ np.linalg.inv(Toff)) @ np.stack([DX + OX, DY + OY, np.ones(DX.size)])
        du = q[0] / q[2] * S; dv = q[1] / q[2] * S
        din_ = (du > -10) & (du < W + 10) & (dv > -10) & (dv < H + 10)
        docc = tree_d[np.clip(dv.astype(int), 0, H - 1), np.clip(du.astype(int), 0, W - 1)]
        dwt = DW_ * (0.75 + 0.25 * np.sin(2 * np.pi * (DFQ * t + DPH))) * np.clip(1 - docc * 1.3, 0, 1) * din_
        U.append(du); V.append(dv); C.append(DCOL); Wt.append(dwt)
        # rim brightness follows the glow just behind the silhouette (backlit by the Milky Way)
        gl_ = cv2.resize(gm, (W, H), interpolation=cv2.INTER_AREA) / 255.0
        gb_ = cv2.GaussianBlur(np.clip((gl_ - FLOOR) / LREF, 0, 1), (0, 0), 8)
        rl = gb_[np.clip(ey_.astype(int), 0, H - 1), np.clip(ex_.astype(int), 0, W - 1)]
        rw_ = RW_ * (0.25 + 0.75 * rl) * (0.8 + 0.2 * np.sin(2 * np.pi * (0.2 * t + RPH))) * float(os.environ.get('RIM_K', '0.12'))
        U.append(ex_); V.append(ey_); C.append(DS.L.PAL[np.full(ex_.size, 215)]); Wt.append(rw_)
        dep = np.zeros_like(dep); lum = np.zeros_like(lum)             # no dot sheet: only particles
    keep = np.concatenate(Wt) > 1e-4
    DS.EXTRA.update(u=np.concatenate(U)[keep], v=np.concatenate(V)[keep], w=np.concatenate(Wt)[keep],
                    col=np.concatenate(C)[keep], z=DS.Z0 - DS.REL * 0.55)
    wr.write(DS.render(dep, 1, t, None, None, None, ST, lum.astype(np.float32)))
    if os.environ.get('ONE') and i >= int(os.environ['ONE']):
        break
wr.close()
print('done', out)
