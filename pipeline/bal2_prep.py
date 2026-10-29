"""Second ballerina: white studio wall -> black, and relight her so she reads like the first ballerina (a pale,
side-lit figure on black). Output: a 'footage' video on black + it is then run through depth_sheet_clip.py with the
first ballerina's preset.
  matte   : depth (figure clearly nearer than the wall; the wall's own depth gradient fitted and removed) gives WHAT is
            her; colour distance to a fitted wall-colour model gives the fine edge (hair, fingers)
  relight : her clothes are black here (the first wore white): the light comes from the depth relief (side key +
            rim), her own skin tones and the skirt's pattern add detail on top
usage: python3 bal2_prep.py src_1920.mp4 depthdir out.mp4 [f0 f1]"""
import sys, os
import numpy as np, cv2
EV = lambda k, d: float(os.environ.get(k, d))
src, dd, out = sys.argv[1], sys.argv[2], sys.argv[3]
cap = cv2.VideoCapture(src); N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
f0, f1 = (int(sys.argv[4]), int(sys.argv[5])) if len(sys.argv) > 5 else (0, N)
for _ in range(f0):
    cap.grab()
import subprocess
OW, OH = 1920, 1080
p = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{OW}x{OH}', '-r', '25', '-i', '-',
                      '-c:v', 'libx264', '-crf', '10', '-preset', 'veryfast', '-pix_fmt', 'yuv420p', out], stdin=subprocess.PIPE)
yy, xx = np.mgrid[0:270, 0:480].astype(np.float32); xn = xx / 480 - 0.5; yn = yy / 270 - 0.5
B = np.stack([np.ones_like(xn), xn, yn, xn * xn, xn * yn, yn * yn], -1)
nv = lambda *a: np.array(a, np.float32) / np.linalg.norm(a)
Lk = nv(-0.75, -0.35, 0.55)                       # key from the left (as the first ballerina's side light)
ST = {}
for i in range(f0, f1):
    ok, fr = cap.read()
    if not ok:
        break
    d = cv2.imread(f'{dd}/d{i:04d}.png', -1).astype(np.float32) / 65535
    d = cv2.resize(d, (OW, OH), interpolation=cv2.INTER_LINEAR)
    f32 = fr.astype(np.float32) / 255
    sm = cv2.resize(f32, (480, 270), interpolation=cv2.INTER_AREA); ds = cv2.resize(d, (480, 270), interpolation=cv2.INTER_AREA)
    # wall = bright, unsaturated, far: robust quadratic fits of its colour and its depth
    lum_s = sm.mean(-1); sat = sm.max(-1) - sm.min(-1)
    msk = (lum_s > np.percentile(lum_s, 35)) & (sat < 0.08)
    for _ in range(3):
        cf = [np.linalg.lstsq(B[msk], sm[..., c][msk], rcond=None)[0] for c in range(3)]
        cd = np.linalg.lstsq(B[msk], ds[msk], rcond=None)[0]
        fit = np.stack([B @ c_ for c_ in cf], -1); dfit = B @ cd
        r_ = np.sqrt(((sm - fit) ** 2).sum(-1)); msk = msk & (r_ < max(np.percentile(r_[msk], 90), 0.01)) & (ds - dfit < 0.08)
    wall = np.stack([cv2.resize((B @ c_).astype(np.float32), (OW, OH), interpolation=cv2.INTER_CUBIC) for c_ in cf], -1)
    dwall = cv2.resize((B @ cd).astype(np.float32), (OW, OH), interpolation=cv2.INTER_CUBIC)
    dk = d - dwall
    near = np.clip((cv2.GaussianBlur(dk, (0, 0), 2) - EV('B_D0', '0.10')) / EV('B_DW', '0.10'), 0, 1)
    dist = np.sqrt(((cv2.GaussianBlur(f32, (0, 0), 0.8) - wall) ** 2).sum(-1))
    c0, c1 = EV('B_C0', '0.06'), EV('B_C1', '0.16')
    al = np.clip((dist - c0) / (c1 - c0), 0, 1); al = al * al * (3 - 2 * al)
    zone = cv2.GaussianBlur(cv2.dilate((near > 0.5).astype(np.uint8), np.ones((25, 25), np.uint8)).astype(np.float32), (0, 0), 4)
    core = cv2.erode((near > 0.5).astype(np.uint8), np.ones((9, 9), np.uint8)).astype(np.float32)
    M = np.clip(np.maximum(al * zone, cv2.GaussianBlur(core, (0, 0), 2)), 0, 1)
    M = cv2.GaussianBlur(M, (0, 0), 0.9)
    # ---- relight ----
    dl = cv2.GaussianBlur(np.where(M > 0.3, dk, 0).astype(np.float32), (0, 0), 6)
    gy, gx = np.gradient(dl)
    k = EV('B_NK', '220')
    nx, ny = -gx * k, -gy * k; nl = np.sqrt(nx * nx + ny * ny + 1)
    lam = np.clip((nx * Lk[0] + ny * Lk[1] + Lk[2]) / nl, 0, 1)
    edge = np.clip(M * cv2.GaussianBlur(1 - M, (0, 0), 6) * 3, 0, 1)                      # wrap / rim
    g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
    skin = np.clip((g - 0.35) / 0.5, 0, 1)                                                     # her skin is the light part
    det = np.clip((g - cv2.GaussianBlur(g, (0, 0), 3)) * 3, -0.4, 0.4)                          # fabric / pattern detail
    cloth = np.clip((0.40 - cv2.GaussianBlur(g, (0, 0), 2)) / 0.22, 0, 1)                   # her black clothes = the glowing garment
    form = (EV('B_FILL', '0.08') + EV('B_KEY', '0.92') * lam ** 1.4) ** EV('B_CON', '1.5')
    alb = EV('B_CLOTH', '1.0') * cloth + EV('B_SKIN', '0.62') * (1 - cloth) * (0.3 + 0.7 * skin)
    Lr = np.clip(form * alb + EV('B_DET', '0.25') * det * cloth + EV('B_RIM', '0.35') * edge * (0.4 + 0.6 * lam), 0, 1)
    Lr = Lr * M
    # warm cream tint (like the first ballerina's dress under her light), on black
    outp = np.stack([Lr * 0.86, Lr * 0.93, Lr * 1.0], -1)
    p.stdin.write((np.clip(outp, 0, 1) * 255 + 0.5).astype(np.uint8).tobytes())
    if os.environ.get('B_DBG') and i == f0:
        cv2.imwrite('/tmp/bal2_dbg_M.jpg', (M * 255).astype(np.uint8)); cv2.imwrite('/tmp/bal2_dbg_L.jpg', (Lr * 255).astype(np.uint8))
    print(i, flush=True)
p.stdin.close(); p.wait()
