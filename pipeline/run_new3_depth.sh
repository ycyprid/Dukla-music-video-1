set -e
cd /home/claude/mv
# name : source : start : n  (10-frame margins for the smoothing)
for spec in "leaf:/mnt/user-data/attachments/f42362b3-e77e-5554-9434-580f66a1e90e/0b8ccd2a-falling_leaf_1.mp4:10:50" "fireworks:/mnt/user-data/attachments/f42362b3-e77e-5554-9434-580f66a1e90e/883e5634-fireworks.mp4:50:50" "water:/mnt/user-data/attachments/f42362b3-e77e-5554-9434-580f66a1e90e/21c78dbe-8348420-hd_2048_1080_25fps.mp4:25:50"; do
  IFS=: read nm f st n <<<"$spec"
  m0=$((st>10 ? st-10 : 0)); pre=$((st-m0)); tot=$((n+pre+10))
  crop=""; [ $nm = water ] && crop="crop=1920:1080:64:0,"
  ffmpeg -v error -y -i "$f" -vf "select='between(n,$m0,$((m0+tot-1)))',setpts=N/25/TB,${crop}scale=1280:720:flags=area" -r 25 -an -c:v libx264 -preset veryfast -crf 12 -pix_fmt yuv420p depthjobs/${nm}_t.mp4
  rm -rf depthjobs/${nm}_traw depthjobs/${nm}_ts out/depth_${nm}_test; mkdir -p out/depth_${nm}_test
  python3 depth_full2.py depthjobs/${nm}_t.mp4 depthjobs/${nm}_traw 100000 0 640 360 > /dev/null
  python3 depth_stab2.py depthjobs/${nm}_t.mp4 depthjobs/${nm}_traw depthjobs/${nm}_ts 4 0.04 0.25 1280 720 > /dev/null
  for ((i=0;i<n;i++)); do cp depthjobs/${nm}_ts/$(printf d%04d.png $((i+pre))) out/depth_${nm}_test/$(printf d%04d.png $i); done
  echo "depth $nm done $(date +%T)"
done
