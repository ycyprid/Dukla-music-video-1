import cv2,numpy as np,glob,sys
d=sys.argv[1]
ims=[cv2.resize(cv2.imread(f),(640,360)) for f in sorted(glob.glob(f'out/stills_{d}/*.png'))]
cv2.imwrite(f'out/sheet{d}.jpg',np.hstack(ims))
