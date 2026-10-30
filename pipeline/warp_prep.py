"""Light-streak warp -> overlay plate: floor cut + curve, then a per-ray brightness field (noise in the angle around the
vanishing point, constant along each ray) so every streak keeps its own brightness as it flies out: a few bright,
most dim. usage: python3 warp_prep.py src.mp4 out.mp4"""
import sys, os, subprocess
import numpy as np, cv2
EV = lambda k, d: float(os.environ.get(k, d))
src, out = sys.argv[1], sys.argv[2]
cap = cv2.VideoCapture(src)
p = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', '1920x1080', '-r', '25', '-i', '-',
                      '-c:v', 'libx264', '-crf', '8', '-preset', 'veryfast', '-pix_fmt', 'yuv420p', out], stdin=subprocess.PIPE)
yy, xx = np.mgrid[0:1080, 0:1920].astype(np.float32)
cx, cy = EV('W_CX', '960'), EV('W_CY', '560')
th = np.arctan2(yy - cy, xx - cx); rr = np.hypot(xx - cx, yy - cy)
rng = np.random.default_rng(3)
K = [(int(k), rng.uniform(0, 2 * np.pi), rng.uniform(0.5, 1.0) / k ** 0.5) for k in (3, 5, 8, 13, 21, 34, 55)]
fl, ga = EV('W_FLOOR', '0.075'), EV('W_GAM', '1.15')
lo, gp = EV('W_LO', '0.15'), EV('W_GP', '2.2')
i = 0
while True:
    ok, fr = cap.read()
    if not ok:
        break
    t = i / 25.0
    n = sum(a * np.sin(k * th + ph + 0.15 * t * (1 + k % 3)) for k, ph, a in K)
    n = (n - n.min()) / (np.ptp(n) + 1e-6)
    gain = lo + (1 - lo) * n ** gp                                          # skewed: most rays dim, a few bright
    gain = gain * np.clip(rr / EV('W_R0', '260'), 0.35, 1)                  # the dense core a little quieter
    f = fr.astype(np.float32) / 255
    f = np.clip((f - fl) / (1 - fl), 0, 1) ** ga
    f = f * gain[..., None]
    p.stdin.write((np.clip(f, 0, 1) * 255 + 0.5).astype(np.uint8).tobytes()); i += 1
p.stdin.close(); p.wait()
