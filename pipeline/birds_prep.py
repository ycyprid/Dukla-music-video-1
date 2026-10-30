"""Flock of small dark birds on a cloudy sky -> lit birds on black (for depth_sheet_clip with flat depth + USELUM).
key: black-top-hat (local sky minus pixel) -> only small dark shapes survive, smooth cloud bands vanish.
light: matte x (base + a little of the bird's own tone variation), soft rim.
usage: python3 birds_prep.py src.mp4 out.mp4 [f0 n]"""
import sys, os, subprocess
import numpy as np, cv2
EV = lambda k, d: float(os.environ.get(k, d))
src, out = sys.argv[1], sys.argv[2]
f0, n = (int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) > 4 else (0, 10 ** 9)
cap = cv2.VideoCapture(src)
for _ in range(f0):
    cap.grab()
p = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', '1920x1080', '-r', '25', '-i', '-',
                      '-c:v', 'libx264', '-crf', '8', '-preset', 'veryfast', '-pix_fmt', 'yuv420p', out], stdin=subprocess.PIPE)
K = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(EV('B_K', '41')),) * 2)
k = 0
while k < n:
    ok, fr = cap.read()
    if not ok:
        break
    g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
    gb = cv2.GaussianBlur(g, (0, 0), 0.8)
    th = cv2.morphologyEx(gb, cv2.MORPH_CLOSE, K) - gb                      # how much darker than the local sky
    t0, t1 = EV('B_T0', '0.05'), EV('B_T1', '0.16')
    m = np.clip((th - t0) / (t1 - t0), 0, 1); m = m * m * (3 - 2 * m)
    m = cv2.GaussianBlur(m, (0, 0), 0.6)
    dark = np.clip(th / 0.4, 0, 1)                                         # body (darker) vs thin wing tips
    det = np.clip((cv2.GaussianBlur(g, (0, 0), 2) - g) * 3, -0.2, 0.2)
    L = np.clip(m * (EV('B_BASE', '0.72') + 0.28 * dark) + 0.25 * det * m, 0, 1)
    o = np.stack([L * 0.86, L * 0.93, L], -1)
    p.stdin.write((o * 255 + 0.5).astype(np.uint8).tobytes()); k += 1
p.stdin.close(); p.wait()
