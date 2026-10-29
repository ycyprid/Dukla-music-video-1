import sys, os, numpy as np, cv2
sys.path.insert(0, '/home/claude/mv'); os.chdir('/home/claude/mv')
import depth_sheet as DS
from common import W, H
outdir = sys.argv[1]; f0, f1 = int(sys.argv[2]), int(sys.argv[3]); os.makedirs(outdir, exist_ok=True)
SX, SY = (float(x) for x in os.environ['LAND'].split(','))
GL = float(os.environ.get('GLIDE', '10'))            # sheet px per second: the camera glides forward
LOOPT = float(os.environ.get('LOOPT', '15'))
gx0, gy0 = DS.lattice()
for fi in range(f0, f1):
    t = fi / 25.0
    off = GL * t
    sh = off % DS.PITCH_PX                              # dots ride with the ground, the lattice wraps by one pitch
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32); yy = yy - off
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
    ph_ = 2 * np.pi * t / LOOPT
    h = (field(SC, 3) * np.cos(ph_) + field(SC, 13) * np.sin(ph_)) + 0.25 * (field(SC * 0.45, 4) * np.cos(ph_ * 2) + field(SC * 0.45, 14) * np.sin(ph_ * 2))
    h = (h - np.percentile(h, 1)) / (np.percentile(h, 99.5) - np.percentile(h, 1)); h = np.clip(h, 0, 1); h = h * h * (3 - 2 * h)
    # ---- surface detail: a fine silk-like ripple (anisotropic, warped) + fine grain, both tiny in height ----
    DA = float(os.environ.get('DET_A', '0.035'))
    warp = field(2.5, 9) * 0.35
    silk = np.sin(2 * np.pi * (xw * 0.35 + yw * 1.0) / float(os.environ.get('DET_WL', '0.55')) + warp * 4 - 0.35 * t)
    grain = field(1.2, 11)
    hd = np.clip(h + DA * (0.7 * silk + 0.3 * grain), 0, 1)
    # ---- lighting in world space ----
    REL_ = float(os.environ.get('REL', '0.3'))
    px_x = SX / H; px_y = SY / H                                   # world units per sheet pixel
    hs = cv2.GaussianBlur(hd, (0, 0), 0.8) * REL_
    gy_, gx_ = np.gradient(hs)
    dzdx = gx_ / px_x; dzdy = -gy_ / px_y                            # +y world = into the distance (image up)
    n = np.stack([-dzdx, -dzdy, np.ones_like(dzdx)], -1); n /= np.linalg.norm(n, axis=-1, keepdims=True)
    def nv(*a): a = np.array(a, float); return a / np.linalg.norm(a)
    Lk = nv(-0.6, -0.35, 0.72)                                       # gold key: front-left, raking
    Lb = nv(0.25 + 0.15 * np.sin(2 * np.pi * t / LOOPT), 0.95, 0.35)                                        # back light: behind the terrain -> sheen on crests
    Vc = nv(0, -0.75, 0.66)                                          # towards the camera
    lam = np.clip(n @ Lk, 0, 1)
    Hb = nv(*(Lb + Vc)); spec = np.clip(n @ Hb, 0, 1) ** float(os.environ.get('SPEC_P', '28'))
    Hk = nv(*(Lk + Vc)); spk = np.clip(n @ Hk, 0, 1) ** 60
    fill = float(os.environ.get('FILL', '0.16'))
    lum = fill + float(os.environ.get('KEY', '0.78')) * lam ** 1.25 + float(os.environ.get('SHEEN', '0.75')) * spec + 0.35 * spk
    # cavity: valleys a touch darker (soft occlusion), crests a touch brighter
    cav = hs - cv2.GaussianBlur(hs, (0, 0), 25); lum = lum * np.clip(1 + cav / (REL_ * 0.25 + 1e-6) * 0.6, 0.55, 1.3)
    lum = lum.astype(np.float32)
    img = DS.render(hd.astype(np.float32), 1, t, (gx0, gy0 + sh), None, None, None, lum)
    cv2.imwrite(f'{outdir}/f{fi:04d}.png', (np.clip(img, 0, 1) * 255).astype(np.uint8))
    print(fi, flush=True)
