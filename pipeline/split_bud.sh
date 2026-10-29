set -e
SRC=/mnt/user-data/uploads/12860602_3840_2160_25fps_1.mp4
for s in 1 2; do
  if [ $s = 1 ]; then A=0; B=853; else A=854; B=1592; fi
  # footage: 4K original (frame-accurate) + 1280 working copy
  ffmpeg -y -loglevel error -i $SRC -vf "select='between(n,$A,$B)',setpts=N/25/TB" -r 25 -c:v libx264 -crf 14 -preset veryfast -pix_fmt yuv420p out/bud_shot$s.mp4
  ffmpeg -y -loglevel error -i depthjobs/bud.mp4 -vf "select='between(n,$A,$B)',setpts=N/25/TB" -r 25 -c:v libx264 -crf 16 -preset veryfast depthjobs/bud_shot$s.mp4
  # raw depth, renumbered from 0
  rm -rf depthjobs/bud_shot${s}_raw; mkdir -p depthjobs/bud_shot${s}_raw
  for ((i=A;i<=B;i++)); do ln -s ../bud_raw/$(printf d%04d.png $i) depthjobs/bud_shot${s}_raw/$(printf d%04d.png $((i-A))); done
  python3 depth_stab_full.py depthjobs/bud_shot$s.mp4 depthjobs/bud_shot${s}_raw depthjobs/bud_shot${s}_s1 4 0.04 640 360 > /dev/null
  python3 depth_stab_full.py depthjobs/bud_shot$s.mp4 depthjobs/bud_shot${s}_s1 out/depth_bud_shot$s 5 0.06 1920 1080 > /dev/null
  ffmpeg -y -loglevel error -framerate 25 -i out/depth_bud_shot$s/d%04d.png -vf scale=1280:720,format=yuv420p -c:v libx264 -crf 18 -preset veryfast out/depth_bud_shot${s}_preview.mp4
  echo shot $s done $(ls out/depth_bud_shot$s | wc -l)
done
