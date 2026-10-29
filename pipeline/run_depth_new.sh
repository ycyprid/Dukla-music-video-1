set -e
cd /home/claude/mv
A=/mnt/user-data/attachments/f42362b3-e77e-5554-9434-580f66a1e90e
DAN=$A/8ba4048d-8462446-hd_2048_1080_25fps.mp4
PIG=$A/b264125e-12772675_1920_1080_25fps.mp4
echo "== dancer $(date +%T)"
# 2.6 s .. 18.4 s (frames 65..459, 10-frame margins for the smoothing), centre crop 2048 -> 1920
ffmpeg -v error -y -i $DAN -vf "select='between(n,65,459)',setpts=N/25/TB,crop=1920:1080:64:0,scale=1280:720:flags=area" -r 25 -an -c:v libx264 -preset veryfast -crf 12 -pix_fmt yuv420p depthjobs/dancer_m.mp4
# delivered footage cut 3-18 s (frames 75..449), 1920x1080 centre crop, for AE
ffmpeg -v error -y -i $DAN -vf "select='between(n,75,449)',setpts=N/25/TB,crop=1920:1080:64:0" -r 25 -an -c:v libx264 -preset veryfast -crf 14 -pix_fmt yuv420p out/dancer_3-18s_1920.mp4
python3 depth_full.py depthjobs/dancer_m.mp4 depthjobs/dancer_raw 100000 0 640 360 > depthjobs/dancer.log 2>&1
python3 depth_stab_full.py depthjobs/dancer_m.mp4 depthjobs/dancer_raw depthjobs/dancer_s1 4 0.04 640 360 >> depthjobs/dancer.log 2>&1
python3 depth_stab_full.py depthjobs/dancer_m.mp4 depthjobs/dancer_s1 depthjobs/dancer_s2 5 0.06 1920 1080 >> depthjobs/dancer.log 2>&1
rm -rf out/depth_dancer; mkdir -p out/depth_dancer
for ((i=0;i<375;i++)); do mv depthjobs/dancer_s2/$(printf d%04d.png $((i+10))) out/depth_dancer/$(printf d%04d.png $i); done
ffmpeg -v error -y -i depthjobs/dancer_m.mp4 -vf "select='between(n,10,384)',setpts=N/25/TB" -r 25 -c:v libx264 -preset veryfast -crf 12 -pix_fmt yuv420p depthjobs/dancer.mp4
ffmpeg -v error -y -framerate 25 -i out/depth_dancer/d%04d.png -vf "format=gray,scale=1280:720" -c:v libx264 -preset veryfast -crf 16 -pix_fmt yuv420p out/depth_dancer_preview.mp4
echo "done dancer $(ls out/depth_dancer | wc -l) frames $(date +%T)"
echo "== pigeons $(date +%T)"
ffmpeg -v error -y -i $PIG -vf scale=1280:720:flags=area -an -c:v libx264 -preset veryfast -crf 12 -pix_fmt yuv420p depthjobs/pigeons.mp4
python3 depth_full.py depthjobs/pigeons.mp4 depthjobs/pigeons_raw 100000 0 640 360 > depthjobs/pigeons.log 2>&1
python3 depth_stab_full.py depthjobs/pigeons.mp4 depthjobs/pigeons_raw depthjobs/pigeons_s1 4 0.04 640 360 >> depthjobs/pigeons.log 2>&1
python3 depth_stab_full.py depthjobs/pigeons.mp4 depthjobs/pigeons_s1 out/depth_pigeons 5 0.06 1920 1080 >> depthjobs/pigeons.log 2>&1
ffmpeg -v error -y -framerate 25 -i out/depth_pigeons/d%04d.png -vf "format=gray,scale=1280:720" -c:v libx264 -preset veryfast -crf 16 -pix_fmt yuv420p out/depth_pigeons_preview.mp4
echo "done pigeons $(ls out/depth_pigeons | wc -l) frames $(date +%T)"
