#!/bin/bash
# Pack the render inputs into data_archive/ (lossless): PNG sequences -> FFV1 .mkv split into 95 MB parts.
# usage: tools/archive_pack.sh <WORK dir (has out/, depthjobs/)> <ATTACH dir> <dest data_archive dir>
set -e
W=$1; A=$2; D=$3; mkdir -p $D/seq $D/plates $D/sources $D/meshes
pack() {  # dir prefix pixfmt
  local d=$1 p=$2 pf=$3 n=$(basename $1)
  [ -f $D/seq/$n.done ] && return
  ffmpeg -v error -y -framerate 25 -i $d/${p}%04d.png -c:v ffv1 -level 3 -pix_fmt $pf -slices 4 /tmp/$n.mkv
  split -b 95M -d -a 2 /tmp/$n.mkv $D/seq/$n.mkv.part; rm /tmp/$n.mkv
  echo "$n $(ls $d/${p}*.png | wc -l) $p $pf" > $D/seq/$n.done
}
for n in depth_blossom2s depth_pigeons_v3 depth_dancer_v3 depth_leaf depth_hands depth_water depth_fireworks depth_rose depth_leaf3 depth_balf; do
  pack $W/out/$n d gray16le; done
pack $W/out/blossom_matte4 m gray; cp $W/out/blossom_matte4/change.npy $D/seq/blossom_matte4_change.npy
for f in out/blossom_footage_v2.mp4 depthjobs/dancer_g75.mp4 depthjobs/balf_full_g.mp4 depthjobs/refl1_fc.mp4 depthjobs/refl2_f.mp4 \
         out/fireworks_footage_25fps_1920.mp4 out/hands_footage_25fps_1920.mp4 out/leaf3_footage_25fps_1920.mp4 \
         out/leaf_footage_25fps_1920.mp4 out/water_footage_25fps_1920.mp4 out/dancer_3-18s_1920.mp4; do
  cp -n $W/$f $D/plates/; done
for f in 60b9fb00-snow.mp4 b264125e-12772675_1920_1080_25fps.mp4 e2159d18-coupleshot_2.mp4 02f87d17-7675962-hd_1920_1080_25fps.mp4 \
         9d8441a2-fire.mp4 6e565724-rosa.obj 0ac032c5-rosa.obj.mtl 7e7846bc-image.jpg 5e893272-Red_rose_SF.obj d4fc7877-Red_rose_SF.mtl \
         a2ae3c42-image.jpg 50b48692-image.jpg d4d39c86-image.jpg 38a55e86-reflections.mp4 66d8c8f1-reflections2.mp4 ef68eb02-ballerina_fix.mp4 \
         d9b9682d-7437640-uhd_4096_1974_30fps.mp4 0b8ccd2a-falling_leaf_1.mp4 692c292a-falling_leaf_3.mp4 21c78dbe-8348420-hd_2048_1080_25fps.mp4 \
         883e5634-fireworks.mp4 8ba4048d-8462446-hd_2048_1080_25fps.mp4; do cp -n $A/$f $D/sources/; done
cp -n /tmp/rosa_mesh.npz /tmp/redrose_mesh.npz $D/meshes/
echo packed
