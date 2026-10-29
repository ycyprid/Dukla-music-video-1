"""Stabilise a handheld shot and calm its light flicker, for the dot-sheet renderer (streams: low memory).
  1. camera path: per-frame similarity (features + RANSAC = the dominant scene motion), chained; smoothed with a
     Gaussian (keeps the slow move, removes the jitter); every frame warped onto the smooth path
  2. the existing depth maps get the very same warp (no new depth pass)
  3. light: each frame averaged with its flow-warped neighbours (+-K, Gaussian in time) where forward/backward flow
     agree, so glints glow and fade instead of flickering frame to frame
usage: python3 branch_stab.py src.mp4 depthdir out.mp4 outdepthdir"""
import sys, os, subprocess
from collections import deque
import numpy as np, cv2
from scipy.ndimage import gaussian_filter1d

src, dd, out, od = sys.argv[1:5]
SIG = float(os.environ.get('STAB_SIGMA', '22'))        # frames: path smoothing
K = int(os.environ.get('LUM_K', '2'))                   # light smoothing half-window (frames)
LS = float(os.environ.get('LUM_SIGMA', '1.2'))          # Gaussian (frames) for the light smoothing
os.makedirs(od, exist_ok=True)
SW, SH = 960, 540


def frames(p):
    cap = cv2.VideoCapture(p)
    while True:
        ok, f = cap.read()
        if not ok:
            return
        yield f


# ---- 1. camera path ----
G = []
for f in frames(src):
    H0, W0 = f.shape[:2]
    G.append(cv2.cvtColor(cv2.resize(f, (SW, SH), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY))
N = len(G); s = W0 / SW
P = [np.eye(3)]
for i in range(N - 1):
    p = cv2.goodFeaturesToTrack(G[i], 800, 0.01, 7)
    q, st, _ = cv2.calcOpticalFlowPyrLK(G[i], G[i + 1], p, None, winSize=(21, 21), maxLevel=3)
    ok_ = st.ravel() == 1
    A, _ = cv2.estimateAffinePartial2D(p[ok_], q[ok_], method=cv2.RANSAC, ransacReprojThreshold=1.5)
    if A is None:
        A = np.array([[1, 0, 0], [0, 1, 0]], float)
    A = A.copy(); A[:, 2] *= s                                     # to full-res px
    P.append(np.vstack([A, [0, 0, 1]]) @ P[-1])                   # frame-0 coords -> frame-i coords
Tc = np.array([[1, 0, W0 / 2], [0, 1, H0 / 2], [0, 0, 1.0]]); Tci = np.linalg.inv(Tc)
Pc = [Tci @ M @ Tc for M in P]                                     # about the frame centre (rotation decoupled)
prm = np.array([[M[0, 2], M[1, 2], np.arctan2(M[1, 0], M[0, 0]), np.log(np.hypot(M[0, 0], M[1, 0]))] for M in Pc])
sm = gaussian_filter1d(prm, SIG, axis=0, mode='nearest')


def mk(tx, ty, a, ls):
    c, s_ = np.cos(a) * np.exp(ls), np.sin(a) * np.exp(ls)
    return np.array([[c, -s_, tx], [s_, c, ty], [0, 0, 1.0]])


C = [Tc @ mk(*sm[i]) @ np.linalg.inv(mk(*prm[i])) @ Tci for i in range(N)]   # frame i -> smooth path
d_ = prm - sm
print('jitter removed: max px %.1f %.1f, rot max %.3f deg, scale max %.4f' % (np.abs(d_[:, 0]).max(), np.abs(d_[:, 1]).max(),
      np.degrees(np.abs(d_[:, 2]).max()), np.abs(d_[:, 3]).max()), flush=True)
np.save(out.replace('.mp4', '_C.npy'), np.array(C))
# ---- 2. warp footage (intermediate, near-lossless) + depth ----
tmp = out.replace('.mp4', '_warp.mkv')
pw = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{W0}x{H0}', '-r', '25', '-i', '-',
                       '-c:v', 'ffv1', tmp], stdin=subprocess.PIPE)
for i, f in enumerate(frames(src)):
    M = C[i][:2]
    pw.stdin.write(cv2.warpAffine(f, M, (W0, H0), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT101).tobytes())
    dp = cv2.imread(f'{dd}/d{i:04d}.png', -1)
    if dp.shape[1] != W0:
        Sd = np.diag([dp.shape[1] / W0, dp.shape[0] / H0, 1.0]); M = (Sd @ C[i] @ np.linalg.inv(Sd))[:2]
    cv2.imwrite(f'{od}/d{i:04d}.png', cv2.warpAffine(dp, M, (dp.shape[1], dp.shape[0]), flags=cv2.INTER_LINEAR,
                                                     borderMode=cv2.BORDER_REFLECT101))
pw.stdin.close(); pw.wait()
print('warped', flush=True)
# ---- 3. light smoothing along the motion (rolling window of 2K+1 frames) ----
dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
gx, gy = np.meshgrid(np.arange(SW, dtype=np.float32), np.arange(SH, dtype=np.float32))
po = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{W0}x{H0}', '-r', '25', '-i', '-',
                       '-c:v', 'libx264', '-crf', '10', '-preset', 'veryfast', '-pix_fmt', 'yuv420p', out], stdin=subprocess.PIPE)
buf = {}; it = frames(tmp); nxt = 0
def get(j):
    global nxt
    while nxt <= j:
        f = next(it); buf[nxt] = (f, cv2.cvtColor(cv2.resize(f, (SW, SH), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)); nxt += 1
    for k in [k for k in buf if k < j - 2 * K - 1]:
        del buf[k]
    return buf[j]
for i in range(N):
    fi, gi = get(min(i + K, N - 1)) and get(i)
    acc = fi.astype(np.float32); wsum = np.ones((H0, W0, 1), np.float32)
    for k in range(-K, K + 1):
        j = i + k
        if k == 0 or j < 0 or j >= N:
            continue
        fj, gj = buf[j]
        fw = dis.calc(gi, gj, None); bw = dis.calc(gj, gi, None)
        bx = cv2.remap(bw[..., 0], gx + fw[..., 0], gy + fw[..., 1], cv2.INTER_LINEAR)
        by = cv2.remap(bw[..., 1], gx + fw[..., 0], gy + fw[..., 1], cv2.INTER_LINEAR)
        err = np.hypot(fw[..., 0] + bx, fw[..., 1] + by)
        cw = np.exp(-(err / 0.8) ** 2) * np.exp(-0.5 * (k / LS) ** 2)
        mx = cv2.resize((gx + fw[..., 0]) * s, (W0, H0)); my = cv2.resize((gy + fw[..., 1]) * s, (W0, H0))
        wj = cv2.resize(cw.astype(np.float32), (W0, H0))[..., None]
        acc += cv2.remap(fj, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101).astype(np.float32) * wj
        wsum += wj
    po.stdin.write(np.clip(acc / wsum + 0.5, 0, 255).astype(np.uint8).tobytes())
    if i % 50 == 0:
        print('light', i, flush=True)
po.stdin.close(); po.wait()
os.remove(tmp)
print('done', out)
