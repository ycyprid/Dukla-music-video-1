"""Flow-guided temporal stabilisation of depth maps.
Neighbouring depth maps (t-K..t+K) are warped into frame t with optical flow, then averaged
with weights that drop where the warp doesn't match (occlusion / fast motion) -> no ghosting.
Only depth is filtered; the colour footage is never blended.
usage: python3 depth_stab.py src.mp4 depthdir outdir [K]"""
import sys, os, glob, cv2, numpy as np
src, dd, od = sys.argv[1], sys.argv[2], sys.argv[3]
K = int(sys.argv[4]) if len(sys.argv) > 4 else 4
ERR = float(sys.argv[5]) if len(sys.argv) > 5 else 0.04
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
yy, xx = np.mgrid[0:FH, 0:FW].astype(np.float32)
sig = K / 2.0
for t in range(N):
    acc = dep[t].copy(); wsum = np.ones_like(acc)
    for o in range(-K, K + 1):
        s = t + o
        if o == 0 or s < 0 or s >= N:
            continue
        flow = cv2.calcOpticalFlowFarneback(gray[t], gray[s], None, 0.5, 4, 21, 3, 5, 1.1, 0)
        mx, my = xx + flow[..., 0], yy + flow[..., 1]
        wd = cv2.remap(dep[s], mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        wg = cv2.remap(gray[s], mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        err = np.abs(wg.astype(np.float32) - gray[t].astype(np.float32)) / 255
        w = np.exp(-(o * o) / (2 * sig * sig)) * np.exp(-(err / ERR) ** 2)
        acc += wd * w; wsum += w
    out = cv2.resize(acc / wsum, (1280, 720), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(f'{od}/d{t:04d}.png', (np.clip(out, 0, 1) * 65535).astype(np.uint16))
    if t % 30 == 0:
        print('stab', t, flush=True)
