import sys, os, numpy as np, cv2
sys.path.insert(0, '/home/claude/mv'); os.chdir('/home/claude/mv')
import depth_sheet as DS
from common import W, H
mode, out = sys.argv[1], sys.argv[2]
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32); xn = xx / W; yn = yy / H
if mode == 'swell':
    rng = np.random.default_rng(3); h = np.zeros((H, W), np.float32)
    for k in range(7):                                   # few broad swells, weak fine detail (k^-2 amplitude)
        ang = rng.uniform(0, np.pi); f = float(os.environ.get("SW_F", "2.4")) * (1.55 ** k); a = f ** -2.4
        h += a * np.sin(2 * np.pi * f * (xn * np.cos(ang) + yn * np.sin(ang) * 0.56) + rng.uniform(0, 6.28))
    h = cv2.GaussianBlur(h, (0, 0), 6); h = (h - h.min()) / np.ptp(h)
    h = h * h * (3 - 2 * h)                              # gradual rise, soft crests
    dep = h
else:
    p = {'petal': 'out/depth_rose/d0130.png', 'ripple': 'out/depth_water/d0200.png'}[mode]
    dep = cv2.resize(cv2.imread(p, -1).astype(np.float32) / 65535, (W, H))
    sg = float(os.environ.get('DETREND', '0'))
    if sg > 0:                                           # keep the relief, drop the broad slope of the plane
        dep = dep - cv2.GaussianBlur(dep, (0, 0), sg)
    dep = np.clip((dep - np.percentile(dep, 2)) / (np.percentile(dep, 99.5) - np.percentile(dep, 2)), 0, 1)
    dep = cv2.GaussianBlur(dep, (0, 0), float(os.environ.get('SMOOTH', '1.5')))
img = DS.render(dep.astype(np.float32), 1, 3.0, None, None, None, None, None)
cv2.imwrite(out, (np.clip(img, 0, 1) * 255).astype(np.uint8))
print('done', out)
