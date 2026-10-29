"""Style frames for new footage: depth from ONE frame (Depth Anything V2 small) + the locked halftone look.
usage: python3 stills_new.py video frame_index mode out.png     mode: leaf | water | fireworks
(look settings come from the environment, as for the other shots)"""
import sys, os
import numpy as np, cv2, onnxruntime as ort
import depth_sheet as DS
from common import W, H

src, fi, mode, out = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
cap = cv2.VideoCapture(src)
for _ in range(fi):
    cap.grab()
ok, fr = cap.read()
# centre-crop wider-than-16:9 footage (2048x1080 -> 1920x1080)
h0, w0 = fr.shape[:2]
if w0 / h0 > 16 / 9 + 0.01:
    cw = int(round(h0 * 16 / 9)); x0 = (w0 - cw) // 2; fr = fr[:, x0:x0 + cw]
frw = cv2.resize(fr, (W, H), interpolation=cv2.INTER_AREA)
cv2.imwrite(out.replace('.png', '_src.jpg'), frw)
# ---- depth for this single frame ----
so = ort.SessionOptions(); so.intra_op_num_threads = 2
sess = ort.InferenceSession('models/depth_anything_v2_vits_dynamic.onnx', so, providers=['CPUExecutionProvider'])
IW, IH = 518, 294
mean = np.array([0.485, 0.456, 0.406], np.float32); std = np.array([0.229, 0.224, 0.225], np.float32)
x = cv2.resize(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB), (IW, IH), interpolation=cv2.INTER_CUBIC).astype(np.float32) / 255
x = ((x - mean) / std).transpose(2, 0, 1)[None]
d = sess.run(None, {sess.get_inputs()[0].name: x})[0].squeeze().astype(np.float32)
d = cv2.resize(d, (W, H), interpolation=cv2.INTER_CUBIC)
d = np.clip((d - np.percentile(d, 2)) / (np.percentile(d, 99.7) - np.percentile(d, 2) + 1e-6), 0, 1)
cv2.imwrite(out.replace('.png', '_depth.png'), (d * 65535).astype(np.uint16))
g = (frw.max(2) if mode == 'fireworks' else cv2.cvtColor(frw, cv2.COLOR_BGR2GRAY)).astype(np.float32) / 255   # sparks: brightest channel (blue counts)
LREF = max(np.percentile(g, 99.5), 0.2)
lum = g / LREF
if mode == 'leaf':
    # subject on black: tight cut-out from the footage (lit or right next to lit), depth for relief
    lit = (cv2.GaussianBlur(g, (0, 0), 1.5) > 0.05).astype(np.uint8)
    lit = cv2.morphologyEx(lit, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    mt = cv2.GaussianBlur(cv2.dilate(lit, np.ones((3, 3), np.uint8)).astype(np.float32), (0, 0), 1.2)
    dd = d * mt
    sel = mt > 0.5
    dep = np.clip(dd / (np.percentile(dd[sel], 95) + 1e-6) * 0.8, 0, 1) if sel.any() else dd
elif mode == 'water':
    # the whole frame is surface: everything present, depth = the receding plane + ripple relief
    dep = 0.30 + 0.70 * d
elif mode == 'scene':
    # full-frame scene (foliage, sky): everything present, light from the footage, depth for relief
    dep = 0.30 + 0.70 * d
elif mode == 'silh':
    # backlit silhouettes against sky: the plants are the subject, the sky goes to black.
    # presence = darker than the sky level; light = backlit rim at the sky edge + the leaves' own texture
    gb = cv2.GaussianBlur(g, (0, 0), 1.0)
    # smooth sky model: quadratic fit to bright, flat pixels (handles a sky that darkens across the frame)
    sm = cv2.resize(gb, (192, 108), interpolation=cv2.INTER_AREA)
    det = np.abs(sm - cv2.GaussianBlur(sm, (0, 0), 2))
    cand = (sm > np.percentile(sm, 60) * 0.9) & (det < 0.04) & (sm > 0.25)
    yy, xx = np.mgrid[0:108, 0:192].astype(np.float32); xx /= 192; yy /= 108
    Bs = np.stack([np.ones_like(xx), xx, yy, xx * xx, xx * yy, yy * yy], -1)
    coef = np.linalg.lstsq(Bs[cand], sm[cand], rcond=None)[0]
    skylo = 0.75 * np.median(sm[cand]); sky = cv2.resize(np.clip(Bs @ coef, skylo, 1.0).astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)
    T = sky * float(os.environ.get('SILH_T', '0.72'))
    m = np.clip((T - gb) / (0.12 * sky), 0, 1)
    m = cv2.GaussianBlur(m, (0, 0), 1.0)
    edge = np.clip(cv2.GaussianBlur(np.abs(cv2.Laplacian(cv2.GaussianBlur(m, (0, 0), 1.5), cv2.CV_32F)), (0, 0), 1.5) * 9, 0, 1)
    rim = edge * m ** 0.3
    hp = np.abs(gb - cv2.GaussianBlur(gb, (0, 0), 4)) / sky
    tex = np.clip(hp * 5, 0, 1)
    own = np.clip(gb / (0.6 * sky), 0, 1) ** 0.9                       # the subject's own light, relative to the sky
    lum = np.clip(m * (float(os.environ.get('SILH_IN', '0.6')) * own + float(os.environ.get('SILH_TEX', '0.45')) * tex)
                  + float(os.environ.get('SILH_RIM', '1.1')) * rim, 0, 1.2)
    dep = m * (0.45 + 0.55 * d)
    LREF = 1.0
elif mode == 'fireworks':
    # sparks in the sky: their own light says where the dots are; depth only adds a faint relief
    pres = np.clip(cv2.GaussianBlur(g, (0, 0), 1.2) / 0.08, 0, 1)
    dep = pres * (0.55 + 0.45 * cv2.GaussianBlur(d, (0, 0), 6))
else:
    raise SystemExit('mode?')
img = DS.render(dep.astype(np.float32), 1, fi / 25.0, None, None, None, None, lum.astype(np.float32))
cv2.imwrite(out, (np.clip(img, 0, 1) * 65535 + 0.5).astype(np.uint16))
print('done', out)
