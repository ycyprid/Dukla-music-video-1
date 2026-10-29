import sys, cv2, subprocess, numpy as np
head, body, out, k = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
H_ = cv2.VideoCapture(head); B_ = cv2.VideoCapture(body)
p = None; i = 0
while True:
    okb, fb = B_.read()
    if not okb: break
    if i < k:
        okh, fh = H_.read(); fb = fh
    if p is None:
        h, w = fb.shape[:2]
        p = subprocess.Popen(['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','bgr24','-s',f'{w}x{h}','-r','25','-i','-','-c:v','libx264','-preset','veryfast','-crf','18','-pix_fmt','yuv420p',out], stdin=subprocess.PIPE)
    p.stdin.write(fb.tobytes()); i += 1
p.stdin.close(); p.wait(); print('frames', i)
