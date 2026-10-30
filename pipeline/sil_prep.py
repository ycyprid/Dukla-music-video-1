"""Backlit silhouette (dark hands on a bright sky) -> a glowing figure on black for the ballerina look.
matte: darkness vs a fitted sky gradient; relight: relief from the matte's distance field (rounded limbs), side key,
strong rim (backlight wrap) + the little tonal detail inside the silhouette.
usage: python3 sil_prep.py src.mp4 out.mp4"""
import sys, os, subprocess
import numpy as np, cv2
EV = lambda k, d: float(os.environ.get(k, d))
src, out = sys.argv[1], sys.argv[2]
cap = cv2.VideoCapture(src); OW, OH = 1920, 1080
p = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{OW}x{OH}', '-r', '25', '-i', '-',
                      '-c:v', 'libx264', '-crf', '10', '-preset', 'veryfast', '-pix_fmt', 'yuv420p', out], stdin=subprocess.PIPE)
yy, xx = np.mgrid[0:270, 0:480].astype(np.float32); xn = xx / 480 - 0.5; yn = yy / 270 - 0.5
B = np.stack([np.ones_like(xn), xn, yn, xn * xn, xn * yn, yn * yn], -1)
nv = lambda *a: np.array(a, np.float32) / np.linalg.norm(a)
Lk = nv(-0.7, -0.45, 0.55)
while True:
    ok, fr = cap.read()
    if not ok: break
    fr = cv2.resize(fr, (OW, OH))
    g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
    gs = cv2.resize(g, (480, 270), interpolation=cv2.INTER_AREA)
    msk = gs > np.percentile(gs, 40)
    for _ in range(3):
        c = np.linalg.lstsq(B[msk], gs[msk], rcond=None)[0]; fit = B @ c
        msk = msk & (gs > fit - 0.08)
    sky = cv2.resize((B @ c).astype(np.float32), (OW, OH), interpolation=cv2.INTER_CUBIC)
    dk = sky - cv2.GaussianBlur(g, (0, 0), 0.8)
    M = np.clip((dk - EV('S_T0', '0.12')) / EV('S_TW', '0.18'), 0, 1); M = M * M * (3 - 2 * M)
    # relief: distance into the silhouette -> rounded height field
    dist = cv2.distanceTransform((M > 0.5).astype(np.uint8), cv2.DIST_L2, 5)
    hgt = np.sqrt(np.clip(dist / EV('S_R', '28'), 0, 1)); hgt = cv2.GaussianBlur(hgt, (0, 0), 3)
    gy, gx = np.gradient(hgt); k = EV('S_NK', '30')
    nx, ny = -gx * k, -gy * k; nl = np.sqrt(nx * nx + ny * ny + 1)
    lam = np.clip((nx * Lk[0] + ny * Lk[1] + Lk[2]) / nl, 0, 1)
    rim = np.clip(M * cv2.GaussianBlur(1 - M, (0, 0), 5) * 3.5, 0, 1)
    inner = np.clip(dk / 0.6, 0, 1)                                 # what tone exists inside (darker = deeper)
    det = np.clip((cv2.GaussianBlur(g, (0, 0), 3) - g) * 4, -0.3, 0.3)
    L = (EV('S_FILL', '0.12') + EV('S_KEY', '0.75') * lam ** 1.3) * (0.75 + 0.25 * inner) + EV('S_DET', '0.3') * det + EV('S_RIM', '0.55') * rim
    L = np.clip(L, 0, 1) * M
    o = np.stack([L * 0.86, L * 0.93, L], -1)
    p.stdin.write((np.clip(o, 0, 1) * 255 + 0.5).astype(np.uint8).tobytes())
p.stdin.close(); p.wait()
