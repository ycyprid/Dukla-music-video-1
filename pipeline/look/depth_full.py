"""Depth Anything V2 (small, ONNX) -> per-frame 16-bit depth PNGs, temporally smoothed.
usage: python3 depth.py src.mp4 outdir nframes [start]"""
import sys, os, time, cv2, numpy as np, onnxruntime as ort
src, outd, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
start = int(sys.argv[4]) if len(sys.argv) > 4 else 0
OW, OH = (int(sys.argv[5]), int(sys.argv[6])) if len(sys.argv) > 6 else (1280, 720)
os.makedirs(outd, exist_ok=True)
so = ort.SessionOptions(); so.intra_op_num_threads = 2
sess = ort.InferenceSession('models/depth_anything_v2_vits_dynamic.onnx', so, providers=['CPUExecutionProvider'])
inp = sess.get_inputs()[0]
IW, IH = 518, 294  # 16:9-ish, multiple of 14
mean = np.array([0.485, 0.456, 0.406], np.float32); std = np.array([0.229, 0.224, 0.225], np.float32)
cap = cv2.VideoCapture(src); cap.set(cv2.CAP_PROP_POS_FRAMES, start)
ema = None; lo = hi = None; T = time.time()
for i in range(n):
    ok, f = cap.read()
    if not ok: break
    x = cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2RGB), (IW, IH), interpolation=cv2.INTER_CUBIC).astype(np.float32) / 255
    x = ((x - mean) / std).transpose(2, 0, 1)[None]
    d = sess.run(None, {inp.name: x})[0].squeeze().astype(np.float32)   # relative inverse depth (bigger = nearer)
    # stable normalisation: slow-moving percentiles so brightness of the depth map doesn't pump
    plo, phi = np.percentile(d, 1), np.percentile(d, 99)
    lo = plo if lo is None else lo * 0.9 + plo * 0.1
    hi = phi if hi is None else hi * 0.9 + phi * 0.1
    dn = np.clip((d - lo) / (hi - lo + 1e-6), 0, 1)
    ema = dn if ema is None else ema * 0.5 + dn * 0.5                 # temporal smoothing against flicker
    out = cv2.resize(ema, (OW, OH), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(f'{outd}/d{i:04d}.png', (np.clip(out, 0, 1) * 65535).astype(np.uint16))
    if i % 30 == 0: print(i, round(time.time() - T, 1), flush=True)
