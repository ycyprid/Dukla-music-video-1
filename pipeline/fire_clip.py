"""Fire in the dot-sheet look: the flame's own light is the picture (no depth model; relief from its blurred light),
embers lift off the top of the flame, transitions: in from below (hands_transition, fdir up), out upwards (odir).
usage: python3 fire_clip.py src.mp4 outdir f0 f1     (frames of the source; CLIPLEN = whole clip)"""
import sys, os
import numpy as np, cv2
import depth_sheet as DS
import common
from common import W, H
common.BEAT = 60.0 / 146
EV = lambda k, d: float(os.environ.get(k, d))
src, od, f0, f1 = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
os.makedirs(od, exist_ok=True)
cap = cv2.VideoCapture(src); NF = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); TL = NF / 25.0
w0 = max(0, f0 - 30)
for _ in range(w0):
    cap.grab()
BT = 60.0 / 146
if os.environ.get('TRANS'):
    DS.TRANS.update(dict(mode='hands', fly_soft=0.8, fdir=(0.0, -1.0), in_xmin=0.0,
                         in0=0.0, in_spread=EV('IN_SP', '2.6') * BT, in_dur=EV('IN_DU', '2.4') * BT,
                         out0=TL - (EV('OUT_SP', '2.2') + EV('OUT_DU', '2.6')) * BT - 0.25,
                         out_spread=EV('OUT_SP', '2.2') * BT, out_dur=EV('OUT_DU', '2.6') * BT,
                         odir=(0.0, -1.0), oscat=0.7, ocx=W / 2, ocy=H / 2))
gx_, gy_ = DS.lattice(); IDS = np.arange(gx_.size, dtype=np.int64)
rng = np.random.default_rng(5)
E = dict(x=np.zeros(0), y=np.zeros(0), vx=np.zeros(0), vy=np.zeros(0), age=np.zeros(0), life=np.zeros(0),
         ph=np.zeros(0), fq=np.zeros(0), c=np.zeros(0), sw=np.zeros(0))
ST = {}; prev = None; dt = 1 / 25.0
for i in range(w0, f1):
    ok, fr = cap.read()
    if not ok:
        break
    frw = cv2.resize(fr, (W, H), interpolation=cv2.INTER_AREA)
    g = (frw.astype(np.float32) @ np.array([0.30, 0.45, 0.25], np.float32)) / 255   # the flame light (warm-weighted luma)
    g = cv2.GaussianBlur(g, (0, 0), 0.7)
    lum = np.clip((g - EV('F_LO', '0.06')) / (EV('F_HI', '0.95') - EV('F_LO', '0.06')), 0, 1) ** EV('F_G', '1.6')
    lum = np.clip(lum + EV('F_DET', '1.4') * (g - cv2.GaussianBlur(g, (0, 0), 4)), 0, 1.1)   # the tongues / wisps
    pres = np.clip(cv2.GaussianBlur(g, (0, 0), 1.2) / EV('F_PT', '0.10'), 0, 1)
    rel = cv2.GaussianBlur(g, (0, 0), 6)
    dep = (pres * (0.45 + 0.55 * np.clip(rel / 0.8, 0, 1))).astype(np.float32)
    # ---- embers: born on the flame's upper edge, rise, drift, flicker, fade ----
    t = i / 25.0
    flow_up = 1.0
    if os.environ.get('TRANS'):                                                   # no embers before / after the flame
        a_ = np.clip((t - 1.6) / 1.0, 0, 1); b_ = np.clip((TL - 1.8 - t) / 0.8, 0, 1); flow_up = a_ * b_
    m8 = (cv2.GaussianBlur(g, (0, 0), 2) > EV('F_ET', '0.35')).astype(np.uint8)
    gy__ = cv2.Sobel(cv2.GaussianBlur(g, (0, 0), 3), cv2.CV_32F, 0, 1)
    top = (cv2.morphologyEx(m8, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0) & (gy__ > 0.01)   # upper edges
    ey, ex = np.nonzero(top)
    k = int(rng.poisson(EV('F_EMB', '3.0') * flow_up)) if len(ex) else 0
    if k:
        # favour the highest edge points (tongue tips)
        wsel = np.exp(-(ey - ey.min()) / 60.0); wsel /= wsel.sum()
        pk = rng.choice(len(ex), k, p=wsel)
        E['x'] = np.r_[E['x'], ex[pk] + rng.uniform(-2, 2, k)]; E['y'] = np.r_[E['y'], ey[pk] + rng.uniform(-2, 2, k)]
        E['vx'] = np.r_[E['vx'], rng.normal(0, 14, k)]; E['vy'] = np.r_[E['vy'], -rng.uniform(70, 160, k)]
        E['age'] = np.r_[E['age'], np.zeros(k)]; E['life'] = np.r_[E['life'], rng.uniform(1.0, 2.4, k)]
        E['ph'] = np.r_[E['ph'], rng.random(k)]; E['fq'] = np.r_[E['fq'], rng.uniform(3, 7, k)]
        E['c'] = np.r_[E['c'], rng.uniform(205, 250, k)]; E['sw'] = np.r_[E['sw'], rng.uniform(-1, 1, k)]
    dr_ = np.exp(-0.6 * dt)
    E['vx'] *= dr_; E['vy'] = E['vy'] * dr_ - 8 * dt                               # hot air keeps lifting them
    E['x'] += (E['vx'] + 25 * E['sw'] * np.sin(2 * np.pi * (0.7 * E['age'] + E['ph']))) * dt; E['y'] += E['vy'] * dt
    E['age'] += dt
    keep = (E['age'] < E['life']) & (E['y'] > -20)
    for k_ in E:
        E[k_] = E[k_][keep]
    a_ = E['age'] / E['life']
    env = (1 - a_) ** 1.3 * (1 - np.exp(-E['age'] / 0.06)) * (0.4 + 0.6 * np.maximum(np.sin(2 * np.pi * (E['fq'] * E['age'] + E['ph'])), 0) ** 4)
    DS.EXTRA.update(u=E['x'], v=E['y'], w=env * EV('F_EMBK', '1.8'), col=DS.L.PAL[np.clip(E['c'] - 60 * a_, 0, 255).astype(np.uint8)])
    img = DS.render(dep, 1, t, None, None, IDS, ST, lum.astype(np.float32))
    if i >= f0:
        cv2.imwrite(f'{od}/fire_{i:04d}.png', (np.clip(img, 0, 1) * 65535 + 0.5).astype(np.uint16)); print(i, flush=True)
