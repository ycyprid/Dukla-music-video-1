set -e
cd /home/claude/mv
rm -rf depthjobs/dancer3_s out/depth_dancer_v3 out/depth_pigeons_v3
python3 depth_stab2.py depthjobs/dancer_m.mp4 depthjobs/dancer2_raw depthjobs/dancer3_s 4 0.04 0.25 1920 1080 > depthjobs/stab3.log 2>&1
mkdir -p out/depth_dancer_v3
for ((i=0;i<375;i++)); do mv depthjobs/dancer3_s/$(printf d%04d.png $((i+10))) out/depth_dancer_v3/$(printf d%04d.png $i); done
echo "done dancer $(date +%T)"
python3 depth_stab2.py depthjobs/pigeons.mp4 depthjobs/pigeons2_raw out/depth_pigeons_v3 4 0.04 0.25 1920 1080 >> depthjobs/stab3.log 2>&1
echo "done pigeons $(date +%T)"
