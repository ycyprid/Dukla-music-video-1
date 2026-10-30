import cv2, numpy as np, subprocess, os

W, H = (int(x_) for x_ in os.environ.get('DESIGN', '1280x720').split('x'))   # design canvas (720x1280 = 9:16)
FPS = 30
BPM = 100.0
BEAT = 60.0 / BPM


def beat(t):
    """Placeholder audio envelope: kick on every beat, accent on bar downbeat, slow 4-bar swell."""
    ph = (t % BEAT) / BEAT
    kick = np.exp(-ph * 7.0)
    bar = int(t / BEAT) % 4
    accent = 1.0 if bar == 0 else 0.6
    swell = 0.5 - 0.5 * np.cos(2 * np.pi * t / (BEAT * 16))
    return float(kick * accent), float(swell)


class Clip:
    def __init__(self, path, maxf=330):
        cap = cv2.VideoCapture(path)
        self.frames = []
        while len(self.frames) < maxf:
            ok, f = cap.read()
            if not ok:
                break
            self.frames.append(f)
        self.n = len(self.frames)

    def get(self, i):
        # ping-pong loop so it never jumps
        m = 2 * self.n - 2
        i = int(i) % m
        if i >= self.n:
            i = m - i
        return self.frames[i]


def gradient_lut(stops):
    """stops: list of (pos, (r,g,b)). returns 256x3 BGR float32 LUT 0..1"""
    xs = np.linspace(0, 1, 256)
    lut = np.zeros((256, 3), np.float32)
    ps = [s[0] for s in stops]
    for c in range(3):
        lut[:, 2 - c] = np.interp(xs, ps, [s[1][c] / 255.0 for s in stops])
    return lut


EMBER = gradient_lut([(0.0, (4, 1, 2)), (0.22, (52, 6, 10)), (0.45, (160, 34, 18)),
                      (0.66, (226, 120, 30)), (0.84, (246, 196, 80)), (1.0, (255, 244, 214))])
AMBER = gradient_lut([(0.0, (6, 2, 1)), (0.3, (90, 22, 8)), (0.6, (214, 112, 26)),
                      (0.85, (250, 205, 110)), (1.0, (255, 250, 232))])


def luma(img):
    return (img[..., 0] * 0.114 + img[..., 1] * 0.587 + img[..., 2] * 0.299)


def grade_map(img, lut, contrast=1.0, pivot=0.45):
    """img float 0..1 BGR -> gradient-mapped"""
    l = luma(img)
    l = np.clip((l - pivot) * contrast + pivot, 0, 1)
    return lut[(l * 255).astype(np.uint8)]


def warm_split(img, amt=0.6):
    """keep footage colour but push shadows red, highlights gold"""
    l = luma(img)[..., None]
    sh = np.array([0.10, 0.12, 0.45], np.float32)  # BGR deep red
    hi = np.array([0.45, 0.82, 1.0], np.float32)   # BGR gold
    tint = sh * (1 - l) + hi * l
    out = img * (1 - amt) + (tint * (0.35 + 0.9 * l)) * amt
    return out


def bloom(img, thresh=0.6, strength=0.6, sigma=18):
    hi = np.clip(img - thresh, 0, None) / (1 - thresh)
    small = cv2.resize(hi, (img.shape[1] // 4, img.shape[0] // 4), interpolation=cv2.INTER_AREA)
    s = sigma / 4
    b = cv2.GaussianBlur(small, (0, 0), s) + cv2.GaussianBlur(small, (0, 0), s * 3) * 0.6
    b = cv2.resize(b, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
    return img + b * strength


def vignette(h=H, w=W, k=0.55):
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    x = (x - w / 2) / (w / 2)
    y = (y - h / 2) / (h / 2)
    r = np.sqrt(x * x * 0.8 + y * y)
    return np.clip(1 - k * r ** 2.2, 0, 1)[..., None].astype(np.float32)


_rng = np.random.default_rng(7)


def grain(img, amt=0.035):
    n = _rng.standard_normal((img.shape[0] // 2, img.shape[1] // 2)).astype(np.float32)
    n = cv2.resize(n, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
    l = luma(np.clip(img, 0, 1))[..., None]
    return img + n[..., None] * amt * (0.4 + l)


def tonemap(img):
    # soft shoulder so highlights roll off instead of clipping (pristine look)
    return 1 - np.exp(-np.clip(img, 0, None) * 1.25) * 1.0 if False else np.clip(img / (1 + np.clip(img - 0.85, 0, None)), 0, 1)


class Writer:
    def __init__(self, path, stills_dir=None, still_frames=()):
        self.path, self.p = path, None                 # ffmpeg starts on the first frame (size taken from it)
        self.stills_dir, self.still_frames, self.i = stills_dir, set(still_frames), 0
        if stills_dir:
            os.makedirs(stills_dir, exist_ok=True)

    def write(self, img):
        if os.environ.get('PNGDIR'):                 # export mode: 16-bit PNG sequence instead of a video
            d_ = os.environ['PNGDIR']; os.makedirs(d_, exist_ok=True)
            cv2.imwrite(os.path.join(d_, f'{os.environ.get("PNGNAME", "f")}_{self.i + int(os.environ.get("PNGOFF", "0")):04d}.png'),
                        (np.clip(img, 0, 1) * 65535 + 0.5).astype(np.uint16))
            self.i += 1
            return
        if self.p is None:
            h_, w_ = img.shape[:2]
            self.p = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
                                       '-s', f'{w_}x{h_}', '-r', str(FPS), '-i', '-', '-c:v', 'libx264',
                                       '-preset', 'veryfast', '-crf', '18', '-pix_fmt', 'yuv420p', self.path],
                                      stdin=subprocess.PIPE)
        if self.i % 30 == 0:
            print('frame', self.i, flush=True)
        u8 = (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)
        self.p.stdin.write(u8.tobytes())
        if self.i in self.still_frames:
            cv2.imwrite(os.path.join(self.stills_dir, f'f{self.i:04d}.png'), u8)
        self.i += 1

    def close(self):
        if self.p is not None:
            self.p.stdin.close()
            self.p.wait()


def soft_pulse(t, attack=0.12, release=0.45):
    """Delicate pulse: smooth rise (attack s) and gentle fall (release s) per beat, no hard edges.
    Stand-in for a smoothed audio envelope (RMS / low band) once the real track is in."""
    ph = t % BEAT
    up = np.clip(ph / attack, 0, 1)
    up = up * up * (3 - 2 * up)
    down = np.exp(-max(0.0, ph - attack) / release)
    bar = int(t / BEAT) % 4
    acc = 1.0 if (bar == 0 or os.environ.get('PULSE_NOACC')) else 0.7
    # blend from the tail of the previous beat so there is no step at the beat boundary
    prev_tail = np.exp(-(ph + BEAT - attack) / release)
    return float(max(up * down, prev_tail * (1 - up)) * acc)


def halftone_color(img, cell, gain=1.15, ramp_lut=None, ramp_mix=0.55):
    h, w = img.shape[:2]
    gw, gh = w // cell, h // cell
    ccol = cv2.resize(img, (gw, gh), interpolation=cv2.INTER_AREA)
    cl = np.clip(luma(ccol) * gain, 0, 1)
    up_l = cv2.resize(cl, (gw * cell, gh * cell), interpolation=cv2.INTER_NEAREST)
    up_c = cv2.resize(ccol, (gw * cell, gh * cell), interpolation=cv2.INTER_NEAREST)
    py, px = (h - gh * cell) // 2, (w - gw * cell) // 2
    up_l = cv2.copyMakeBorder(up_l, py, h - gh * cell - py, px, w - gw * cell - px, cv2.BORDER_CONSTANT)
    up_c = cv2.copyMakeBorder(up_c, py, h - gh * cell - py, px, w - gw * cell - px, cv2.BORDER_CONSTANT)
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    fx = ((x - px) % cell) - cell / 2 + 0.5
    fy = ((y - py) % cell) - cell / 2 + 0.5
    d = np.sqrt(fx * fx + fy * fy)
    r = np.sqrt(up_l) * cell * 0.64
    mask = np.clip(r - d + 0.5, 0, 1)[..., None]
    col = up_c / (luma(up_c)[..., None] + 0.08) * (0.35 + 0.65 * up_l[..., None])
    if ramp_lut is not None:
        ramp = ramp_lut[(up_l * 255).astype(np.uint8)]
        col = col * (1 - ramp_mix) + ramp * ramp_mix * (0.6 + 0.6 * up_l[..., None])
    return col * mask
