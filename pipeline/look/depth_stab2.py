"""Strict flow-guided temporal smoothing of depth (v2). Neighbours t-K..t+K are warped into frame t with optical flow
and averaged only where (a) the warped image matches (photometric check), (b) forward/backward flow agree
(occlusion check) and (c) the warped depth agrees with frame t's own depth (no mixing of different surfaces).
Fast limbs/wings therefore keep only their own frame; slow areas get the full smoothing.
usage: python3 depth_stab2.py src.mp4 depthdir outdir K ERR DERR OW OH"""
import sys, os, glob, cv2, numpy as np
src, dd, od = sys.argv[1], sys.argv[2], sys.argv[3]
K, ERR, DERR = int(sys.argv[4]), float(sys.argv[5]), float(sys.argv[6])
OW, OH = int(sys.argv[7]), int(sys.argv[8])
os.makedirs(od, exist_ok=True)
FW, FH = 640, 360
fs = sorted(glob.glob(f'{dd}/d*.png'))
cap = cv2.VideoCapture(src)
gray, dep = [], []
for f in fs:
    ok, fr = cap.read()
    gray.append(cv2.cvtColor(cv2.resize(fr, (FW, FH), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY))
    dep.append(cv2.resize(cv2.imread(f, -1).astype(np.float32) / 65535, (FW, FH), interpolation=cv2.INTER_AREA))
N = len(fs)
dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
yy, xx = np.mgrid[0:FH, 0:FW].astype(np.float32)
sig = max(K / 2.0, 0.8)
for t in range(N):
    acc = dep[t].copy(); wsum = np.ones_like(acc)
    for o in range(-K, K + 1):
        s = t + o
        if o == 0 or s < 0 or s >= N:
            continue
        fw = dis.calc(gray[t], gray[s], None)
        bw = dis.calc(gray[s], gray[t], None)
        mx, my = xx + fw[..., 0], yy + fw[..., 1]
        wd = cv2.remap(dep[s], mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        wg = cv2.remap(gray[s], mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        bwr = cv2.remap(bw, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        fb = np.hypot(fw[..., 0] + bwr[..., 0], fw[..., 1] + bwr[..., 1])          # ~0 where the flow is consistent
        err = np.abs(wg.astype(np.float32) - gray[t].astype(np.float32)) / 255
        w = np.exp(-(o * o) / (2 * sig * sig)) * np.exp(-(err / ERR) ** 2) * np.exp(-(fb / 1.0) ** 2) \
            * np.exp(-((wd - dep[t]) / DERR) ** 2)
        acc += wd * w; wsum += w
    out = cv2.resize(acc / wsum, (OW, OH), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(f'{od}/d{t:04d}.png', (np.clip(out, 0, 1) * 65535).astype(np.uint16))
    if t % 50 == 0:
        print('stab', t, flush=True)
