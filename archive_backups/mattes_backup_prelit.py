"""Subject mattes for the depth dot-sheet.
depth_matte: subject on black whose depth has a fake glow around it. A smooth surface (cubic polynomial) is fitted to
  the depth where the footage is surely background (dark and far from anything lit) and subtracted; what stands
  clearly above it is the subject -- shadowed hair/back included, since the depth model sees them.
sky_matte: dark subjects on a bright flat sky (luma key against a local sky estimate)."""
import numpy as np, cv2

_SW, _SH = 160, 90
_y, _x = np.mgrid[0:_SH, 0:_SW].astype(np.float32)
_x = _x / _SW - 0.5; _y = _y / _SH - 0.5
_B = np.stack([np.ones_like(_x), _x, _y, _x * _x, _x * _y, _y * _y, _x ** 3, _x * _x * _y, _x * _y * _y, _y ** 3], -1).reshape(-1, 10)


def depth_matte(dep, g8, lift=0.07, soft=0.10):
    H, W = dep.shape
    g = cv2.resize(g8, (W, H)).astype(np.float32) / 255
    lit = (cv2.GaussianBlur(g, (0, 0), 2) > 0.10).astype(np.uint8)
    near = cv2.dilate(lit, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (121, 121)))   # stay well away from her
    bgm = cv2.resize((near == 0).astype(np.uint8), (_SW, _SH), interpolation=cv2.INTER_NEAREST).ravel() > 0
    ds = cv2.resize(dep, (_SW, _SH), interpolation=cv2.INTER_AREA).ravel()
    if bgm.sum() < 200:
        bg = np.full((H, W), np.percentile(dep, 20), np.float32)
    else:
        c, *_ = np.linalg.lstsq(_B[bgm], ds[bgm], rcond=None)
        bg = cv2.resize((_B @ c).reshape(_SH, _SW).astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)
    above = dep - bg
    m = np.clip((above - lift) / soft, 0, 1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    m = m * cv2.GaussianBlur(near.astype(np.float32), (0, 0), 10)          # only around the lit figure
    m = cv2.GaussianBlur(m, (0, 0), 1.5)
    return m, np.clip(above, 0, 1)


def sky_matte(g8, W, H, lo=0.05, soft=0.10):
    g = cv2.resize(g8, (W, H)).astype(np.float32) / 255
    sky = cv2.resize(cv2.dilate(cv2.resize(g, (160, 90), interpolation=cv2.INTER_AREA),
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))), (W, H), interpolation=cv2.INTER_CUBIC)
    sky = cv2.GaussianBlur(sky, (0, 0), 20)
    m = np.clip((sky - g - lo) / soft, 0, 1)
    m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    return cv2.GaussianBlur(m, (0, 0), 1.2)
