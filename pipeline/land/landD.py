import sys, os, numpy as np, cv2
sys.path.insert(0, '/home/claude/mv'); os.chdir('/home/claude/mv')
import depth_sheet as DS
from common import W, H
out = sys.argv[1]; t = float(os.environ.get('T', '3.0'))
SX, SY = (float(x) for x in os.environ['LAND'].split(','))
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
xw = (xx - W / 2) / H * SX; yw = -(yy - H / 2) / H * SY             # world coords of the sheet (y = into the distance)
rng = np.random.default_rng(int(os.environ.get('SEED', '3')))
# organic swells: smooth random field in WORLD units (isotropic on the ground), broad + a weak second octave
def field(scale, seed):
    r_ = np.random.default_rng(seed); gw, gh = 256, 512
    z = r_.normal(size=(gh, gw)).astype(np.float32)
    z = cv2.GaussianBlur(z, (0, 0), scale); z /= z.std() + 1e-6
    # world window -> field texture coords
    fx = ((xw / (SX * W / H) + 0.5) * gw).astype(np.float32); fy = ((yw / SY + 0.5) * gh).astype(np.float32)
    return cv2.remap(z, fx, fy, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT101)
SC = float(os.environ.get('SW_SCALE', '9'))
h = field(SC, 3) + 0.25 * field(SC * 0.45, 4)
h = (h - np.percentile(h, 1)) / (np.percentile(h, 99.5) - np.percentile(h, 1)); h = np.clip(h, 0, 1); h = h * h * (3 - 2 * h)
img = DS.render(h.astype(np.float32), 1, t, None, None, None, None, None)
cv2.imwrite(out, (np.clip(img, 0, 1) * 255).astype(np.uint8)); print('done', out)
