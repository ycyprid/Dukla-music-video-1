set -e
cd /home/claude/mv
echo "== dancer $(date +%T)"
rm -rf depthjobs/dancer2_raw depthjobs/dancer2_s
python3 depth_full2.py depthjobs/dancer_m.mp4 depthjobs/dancer2_raw 100000 0 640 360 > depthjobs/dancer2.log 2>&1
python3 depth_stab2.py depthjobs/dancer_m.mp4 depthjobs/dancer2_raw depthjobs/dancer2_s 3 0.035 0.06 1920 1080 >> depthjobs/dancer2.log 2>&1
rm -rf out/depth_dancer_v2; mkdir -p out/depth_dancer_v2
for ((i=0;i<375;i++)); do mv depthjobs/dancer2_s/$(printf d%04d.png $((i+10))) out/depth_dancer_v2/$(printf d%04d.png $i); done
echo "done dancer $(date +%T)"
echo "== pigeons $(date +%T)"
rm -rf depthjobs/pigeons2_raw
python3 depth_full2.py depthjobs/pigeons.mp4 depthjobs/pigeons2_raw 100000 0 640 360 > depthjobs/pigeons2.log 2>&1
python3 depth_stab2.py depthjobs/pigeons.mp4 depthjobs/pigeons2_raw out/depth_pigeons_v2 3 0.035 0.06 1920 1080 >> depthjobs/pigeons2.log 2>&1
echo "done pigeons $(date +%T)"
