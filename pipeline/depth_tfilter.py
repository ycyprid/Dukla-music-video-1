"""Bidirectional flow-guided recursive average of a depth sequence (same as the matte filter): depth jumps on
moving petals become smooth. usage: depth_tfilter.py footage.mp4 depthdir n outdir   (16-bit, same size as input)"""
import sys, os, numpy as np, cv2
src, dd, n, od = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
os.makedirs(od, exist_ok=True)
A = float(os.environ.get('DT_A', '0.75'))
c = cv2.VideoCapture(src); small = []
for _ in range(n):
    ok, f = c.read(); small.append(cv2.cvtColor(cv2.resize(f, (640, 360), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY))
D = [cv2.resize(cv2.imread(f'{dd}/d{i:04d}.png', -1), (1280, 720), interpolation=cv2.INTER_AREA) for i in range(n)]
H, W = D[0].shape
dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
gxs, gys = np.meshgrid(np.arange(640, dtype=np.float32), np.arange(360, dtype=np.float32))
gx, gy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
def fp(i, j):
    f = dis.calc(small[i], small[j], None); b = dis.calc(small[j], small[i], None)
    bx = cv2.remap(b[..., 0], gxs + f[..., 0], gys + f[..., 1], cv2.INTER_LINEAR); by = cv2.remap(b[..., 1], gxs + f[..., 0], gys + f[..., 1], cv2.INTER_LINEAR)
    tr = np.clip(1.5 - np.hypot(f[..., 0] + bx, f[..., 1] + by), 0, 1)
    return cv2.resize(f, (W, H)) * (W / 640), cv2.resize(tr, (W, H))
def run(order):
    out = {}; prev = None; pi = None
    for i in order:
        Di = D[i].astype(np.float32) / 65535
        if prev is None: cur = Di
        else:
            fl, tr = fp(i, pi)
            w = cv2.remap(prev, gx + fl[..., 0], gy + fl[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            a = A * tr * np.clip(1 - np.abs(w - Di) / 0.35, 0, 1)      # a real depth edge passing: follow the frame
            cur = a * w + (1 - a) * Di
        out[i] = cur.astype(np.float16); prev = cur; pi = i
    return out
F = run(range(n)); B = run(range(n - 1, -1, -1))
for i in range(n):
    S = (F[i].astype(np.float32) + B[i].astype(np.float32)) / 2
    cv2.imwrite(f'{od}/d{i:04d}.png', (np.clip(S, 0, 1) * 65535 + 0.5).astype(np.uint16))
print('done')
