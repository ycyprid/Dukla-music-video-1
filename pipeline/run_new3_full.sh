set -e
cd /home/claude/mv
for spec in "leaf:/mnt/user-data/attachments/f42362b3-e77e-5554-9434-580f66a1e90e/0b8ccd2a-falling_leaf_1.mp4" "fireworks:/mnt/user-data/attachments/f42362b3-e77e-5554-9434-580f66a1e90e/883e5634-fireworks.mp4" "water:/mnt/user-data/attachments/f42362b3-e77e-5554-9434-580f66a1e90e/21c78dbe-8348420-hd_2048_1080_25fps.mp4"; do
  IFS=: read nm f <<<"$spec"
  crop=""; [ $nm = water ] && crop="crop=1920:1080:64:0,"
  # conform to 25 fps frame-for-frame (no interpolation), centre-crop the 2048-wide water
  ffmpeg -v error -y -i "$f" -vf "setpts=N/25/TB,${crop}scale=1280:720:flags=area" -r 25 -an -c:v libx264 -preset veryfast -crf 12 -pix_fmt yuv420p depthjobs/${nm}_full.mp4
  ffmpeg -v error -y -i "$f" -vf "setpts=N/25/TB,${crop}scale=1920:1080:flags=lanczos" -r 25 -an -c:v libx264 -preset veryfast -crf 14 -pix_fmt yuv420p out/${nm}_footage_25fps_1920.mp4
  rm -rf depthjobs/${nm}_fraw out/depth_${nm}
  python3 depth_full2.py depthjobs/${nm}_full.mp4 depthjobs/${nm}_fraw 100000 0 640 360 > /dev/null
  python3 depth_stab2.py depthjobs/${nm}_full.mp4 depthjobs/${nm}_fraw out/depth_${nm} 4 0.04 0.25 1920 1080 > /dev/null
  echo "depth $nm done $(ls out/depth_${nm} | wc -l) frames $(date +%T)"
done
