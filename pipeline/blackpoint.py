"""Lift the renderer's faint background floor (~1.8 % deep red glow of the empty dot sheet) to true black:
x' = (x - BP) / (1 - BP), 16-bit PNGs in place. Same BP for every sequence that must match.
usage: python3 blackpoint.py dir BP16 [start step]"""
import sys, glob, numpy as np, cv2
d, bp = sys.argv[1], float(sys.argv[2])
fs = sorted(glob.glob(f'{d}/*.png'))
s0, st = (int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) > 4 else (0, 1)
for f in fs[s0::st]:
    a = cv2.imread(f, -1).astype(np.float32)
    a = np.clip((a - bp) * 65535.0 / (65535.0 - bp), 0, 65535)
    cv2.imwrite(f, (a + 0.5).astype(np.uint16))
