"""Conform footage with held (duplicate) frames to 25 fps without stutter.
Drops near-duplicate frames, spaces the unique frames evenly over the clip's real duration, then (only if needed)
motion-interpolates to 25 fps. Optional centre-crop to 16:9 and scale to 1920x1080.
usage: python3 conform_dedupe.py src.mp4 out.mp4"""
import sys, subprocess
import numpy as np, cv2

src, out = sys.argv[1], sys.argv[2]
cap = cv2.VideoCapture(src); fps = cap.get(cv2.CAP_PROP_FPS)
frames, small = [], []
while True:
    ok, f = cap.read()
    if not ok:
        break
    h0, w0 = f.shape[:2]
    if w0 / h0 > 16 / 9 + 0.01:
        cw = int(round(h0 * 16 / 9)); x0 = (w0 - cw) // 2; f = f[:, x0:x0 + cw]
    f = cv2.resize(f, (1920, 1080), interpolation=cv2.INTER_AREA)
    frames.append(f); small.append(cv2.cvtColor(cv2.resize(f, (480, 270)), cv2.COLOR_BGR2GRAY).astype(np.float32))
n0 = len(frames); dur = n0 / fps
d = np.array([np.abs(small[i + 1] - small[i]).mean() for i in range(n0 - 1)])
dup = d < 0.2 * np.median(d)
keep = [0] + [i + 1 for i in range(n0 - 1) if not dup[i]]
ufps = len(keep) / dur
print('frames %d at %.3f fps -> %d unique, spaced at %.3f fps' % (n0, fps, len(keep), ufps))
vf = [] if abs(ufps - 25) < 0.05 else ['-vf', 'minterpolate=fps=25:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1']
rate = '25' if not vf else '%.6f' % ufps
p = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', '1920x1080', '-r', rate, '-i', '-']
                     + vf + ['-r', '25', '-c:v', 'libx264', '-crf', '10', '-preset', 'veryfast', '-pix_fmt', 'yuv420p', out], stdin=subprocess.PIPE)
for k in keep:
    p.stdin.write(frames[k].tobytes())
p.stdin.close(); p.wait()
print('done', out)
