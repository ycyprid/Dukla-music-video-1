"""3 s motion: sil_bg3 per frame. usage: motion_sil3.py src start n outdir [render_from render_to]"""
import sys, os, numpy as np, cv2
import common
common.BEAT = 60.0 / 146
import sil_bg3 as S
src, a, n, od = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
r0, r1 = (int(sys.argv[5]), int(sys.argv[6])) if len(sys.argv) > 6 else (a, a + n)
os.makedirs(od, exist_ok=True)
cap = cv2.VideoCapture(src)
w0 = max(a, r0 - 40)
for _ in range(w0):
    cap.grab()
st = {}
for fi in range(w0, r1):
    ok, fr = cap.read()
    img, _ = S.frame(fr, fi / 25.0, st)
    if fi >= r0:
        cv2.imwrite(f'{od}/dance_{fi:04d}.png', (np.clip(img, 0, 1) * 65535 + 0.5).astype(np.uint16))
        print(fi, flush=True)
