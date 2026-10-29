set -e
cd /home/claude/mv
A=/mnt/user-data/attachments/f42362b3-e77e-5554-9434-580f66a1e90e
for pair in "redroses:$A/c1252f85-7675962-hd_1920_1080_25fps.mp4" "peony:$A/c735ef61-13118328-uhd_3840_2160_24fps.mp4" "pinkrose:$A/f39348ac-12142473-hd_1920_1080_25fps.mp4" "bud:/mnt/user-data/uploads/12860602_3840_2160_25fps_1.mp4"; do
  n=${pair%%:*}; f=${pair#*:}
  echo "== $n $(date +%T)"
  ffmpeg -v error -y -i "$f" -vf scale=1280:720:flags=area -an -c:v libx264 -preset veryfast -crf 12 -pix_fmt yuv420p depthjobs/$n.mp4
  fps=$(ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate -of csv=p=0 "$f")
  python3 depth_full.py depthjobs/$n.mp4 depthjobs/${n}_raw 100000 0 640 360 > depthjobs/$n.log 2>&1
  python3 depth_stab_full.py depthjobs/$n.mp4 depthjobs/${n}_raw depthjobs/${n}_s1 4 0.04 640 360 >> depthjobs/$n.log 2>&1
  python3 depth_stab_full.py depthjobs/$n.mp4 depthjobs/${n}_s1 out/depth_$n 5 0.06 1920 1080 >> depthjobs/$n.log 2>&1
  ffmpeg -v error -y -framerate $fps -i out/depth_$n/d%04d.png -vf "format=gray,scale=1280:720" -c:v libx264 -preset veryfast -crf 16 -pix_fmt yuv420p out/depth_${n}_preview.mp4
  echo "done $n $(ls out/depth_$n | wc -l) frames $(date +%T)"
done
