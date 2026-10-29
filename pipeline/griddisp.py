"""Grid displacement test: masked roses (mask from stabilised depth), each grid cell extruded as a block
toward the camera by depth + travelling noise, slight camera turn, our warm palette on black.
usage: python3 griddisp.py still <frame> <out.png>"""
import sys, glob
import cv2, numpy as np
import opensimplex
from common import W, H, luma, warm_split, bloom, vignette, gradient_lut

opensimplex.seed(7)
SRC = 'src/h6b.mp4'
DEP = sorted(glob.glob('depth/roses_b_s2/d*.png'))
CELL = 14
GW, GH = W // CELL, H // CELL
OXp, OYp = (W - GW * CELL) / 2, (H - GH * CELL) / 2
F = 1500.0
EXTRUDE = 75.0
FRAME_C = (830, 430)                                 # reframe on the yellow rose
FRAME_Z = 1.10
vig = vignette(k=0.45)


def read_frame(i):
    cap = cv2.VideoCapture(SRC); cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    return cap.read()[1].astype(np.float32) / 255


def grade(img):
    """our warm look: greens -> amber, warm split, gentle lift"""
    gr = np.clip((img[..., 1] - img[..., 2]) / 0.10, 0, 1)[..., None]
    amber = luma(img)[..., None] * np.array([0.35, 0.85, 1.35], np.float32)
    img = img * (1 - 0.7 * gr) + amber * 0.7 * gr
    g = warm_split(img, 0.45)
    g = np.clip((g - 0.5) * 1.2 + 0.47, 0, 1)
    return np.clip(g, 0, 1) ** 0.85 * 1.12


def mask_from_depth(d):
    m = np.clip((d - 0.42) / 0.14, 0, 1)                    # in-focus near flowers only
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    return cv2.GaussianBlur(m, (0, 0), 3)


def project(x, y, z, yaw, pitch):
    """x,y image-plane px, z = height toward camera (px). Camera turns about the frame centre."""
    xc, yc = x - W / 2, y - H / 2
    c, s = np.cos(yaw), np.sin(yaw)
    x1 = xc * c + z * s; z1 = -xc * s + z * c
    cp, sp = np.cos(pitch), np.sin(pitch)
    y1 = yc * cp + z1 * sp; z2 = -yc * sp + z1 * cp
    k = F / (F - z2)
    return W / 2 + x1 * k, H / 2 + y1 * k, k


def reframe(a):
    M = np.float32([[FRAME_Z, 0, W / 2 - FRAME_Z * FRAME_C[0]], [0, FRAME_Z, H / 2 - FRAME_Z * FRAME_C[1]]])
    return cv2.warpAffine(a, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


def render(i, t=None, yaw=0.07, pitch=-0.05):
    t = i / 30.0 if t is None else t
    img = reframe(grade(read_frame(i)))
    d = reframe(cv2.imread(DEP[min(i, len(DEP) - 1)], -1).astype(np.float32) / 65535)
    m = mask_from_depth(d)
    # per-cell data
    cimg = cv2.resize(img[int(OYp):int(OYp) + GH * CELL, int(OXp):int(OXp) + GW * CELL], (GW, GH), interpolation=cv2.INTER_AREA)
    cd = cv2.resize(d[int(OYp):int(OYp) + GH * CELL, int(OXp):int(OXp) + GW * CELL], (GW, GH), interpolation=cv2.INTER_AREA)
    cm = cv2.resize(m[int(OYp):int(OYp) + GH * CELL, int(OXp):int(OXp) + GW * CELL], (GW, GH), interpolation=cv2.INTER_AREA)
    gx = np.arange(GW); gy = np.arange(GH)
    nz = opensimplex.noise3array(gx * 0.09, gy * 0.09, np.array([t * 0.35]))[0]          # travelling waves
    nz2 = opensimplex.noise3array(gx * 0.35 + 50, gy * 0.35, np.array([t * 0.8]))[0]    # finer block jitter
    # mostly calm relief from depth; travelling clusters of blocks pop out strongly (like the reference)
    pop = np.clip((nz - 0.05) / 0.25, 0, 1); pop = pop * pop * (3 - 2 * pop)
    hgt = EXTRUDE * pop * (0.6 + np.clip((cd - 0.42) / 0.5, 0, 1) * 0.8 + 1.4 * np.maximum(nz2, 0))
    # base: the masked footage on the back plane (unbroken image, no seams), via the plane's homography
    cx = np.array([0, W, W, 0], np.float32); cy = np.array([0, 0, H, H], np.float32)
    bu, bv, _ = project(cx, cy, 0.0, yaw, pitch)
    Hm = cv2.getPerspectiveTransform(np.stack([cx, cy], 1), np.stack([bu, bv], 1).astype(np.float32))
    mm = cv2.resize(cv2.resize(cm, (GW * CELL, GH * CELL), interpolation=cv2.INTER_NEAREST), (W, H))
    mm = cv2.GaussianBlur(np.clip((m - 0.25) / 0.5, 0, 1), (0, 0), 1.5)
    out = cv2.warpPerspective(img * mm[..., None], Hm, (W, H))
    order = np.argsort(hgt.ravel())                          # low blocks first, tall blocks drawn on top
    for idx in order:
        j, ii = divmod(idx, GW)
        a = cm[j, ii]
        h = hgt[j, ii]
        if a < 0.25 or h < 4:                               # flat cells stay part of the base image
            continue
        x0 = OXp + ii * CELL; y0 = OYp + j * CELL
        X = np.array([x0, x0 + CELL, x0 + CELL, x0], np.float32)
        Y = np.array([y0, y0, y0 + CELL, y0 + CELL], np.float32)
        ub, vb, _ = project(X, Y, 0.0, yaw, pitch)
        uf, vf, k = project(X, Y, h, yaw, pitch)
        col = cimg[j, ii] * a
        # sides (the 4 quads between back and front face), shaded darker toward the back
        for e in range(4):
            q = np.array([[ub[e], vb[e]], [ub[(e + 1) % 4], vb[(e + 1) % 4]], [uf[(e + 1) % 4], vf[(e + 1) % 4]], [uf[e], vf[e]]])
            shade = (0.28, 0.20, 0.38, 0.48)[e]
            cv2.fillConvexPoly(out, np.round(q * 4).astype(np.int32), (col * shade).tolist(), cv2.LINE_AA, 2)
        # front face carries the actual footage patch (scaled to its projected size)
        fw = max(int(round(uf[1] - uf[0])), 1); fh = max(int(round(vf[2] - vf[1])), 1)
        patch = img[int(y0):int(y0) + CELL, int(x0):int(x0) + CELL] * a
        patch = cv2.resize(patch, (fw, fh), interpolation=cv2.INTER_LINEAR)
        px, py = int(round(uf[0])), int(round(vf[0]))
        xa, ya = max(px, 0), max(py, 0); xb, yb = min(px + fw, W), min(py + fh, H)
        if xb > xa and yb > ya:
            out[ya:yb, xa:xb] = patch[ya - py:yb - py, xa - px:xb - px]
            # thin bright rim on the front edge (catches light, like the reference)
            rim = np.minimum(col * (1.15 + 0.5 * min(h / (EXTRUDE * 1.5), 1)), 1.0)   # light catching the block edge
            cv2.rectangle(out, (xa, ya), (xb - 1, yb - 1), rim.tolist(), 1, cv2.LINE_AA)
    out = bloom(out * 1.45, 0.6, 0.45, 10) * vig
    Lm = luma(out)[..., None]
    return np.clip(out * ((Lm / (1 + Lm * 0.7)) / np.maximum(Lm, 1e-5)) * 1.15, 0, 1)


if __name__ == '__main__' and sys.argv[1] == 'still':
    i = int(sys.argv[2])
    o = render(i)
    cv2.imwrite(sys.argv[3], (o * 255 + 0.5).astype(np.uint8))
