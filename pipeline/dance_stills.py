import sys, os, numpy as np, cv2
import sil_bg3 as S
src, od = sys.argv[1], sys.argv[2]; fis = [int(x) for x in sys.argv[3].split(',')]
cap = cv2.VideoCapture(src); F = {}
for i in range(max(fis) + 1):
    ok, fr = cap.read()
    if i in fis: F[i] = fr
for i in fis:
    img, _ = S.frame(F[i], i / 25.0, None)
    cv2.imwrite(f'{od}/st_{i:04d}.png', (np.clip(img, 0, 1) * 65535 + 0.5).astype(np.uint16)); print(i, flush=True)
