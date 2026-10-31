#!/bin/bash
# Restore data_archive/ into a work dir: depth/matte sequences back to PNGs, plates to out/ + depthjobs/, meshes to /tmp.
# usage: tools/archive_restore.sh <data_archive dir> <WORK dir>
set -e
D=$1; W=$2; mkdir -p $W/out $W/depthjobs
for f in $D/seq/*.done; do
  read n cnt p pf < $f
  cat $D/seq/$n.mkv.part* > /tmp/$n.mkv; mkdir -p $W/out/$n
  ffmpeg -v error -y -i /tmp/$n.mkv -pix_fmt $pf -start_number 0 $W/out/$n/${p}%04d.png; rm /tmp/$n.mkv
  echo "$n: $(ls $W/out/$n | wc -l) / $cnt"
done
cp $D/seq/blossom_matte4_change.npy $W/out/blossom_matte4/change.npy
for f in dancer_g75 balf_full_g refl1_fc refl2_f; do cp $D/plates/$f.mp4 $W/depthjobs/; done
for f in $D/plates/*.mp4; do b=$(basename $f); case $b in dancer_g75*|balf_full_g*|refl1_fc*|refl2_f*) ;; *) cp $f $W/out/;; esac; done
cp $D/meshes/*.npz /tmp/
echo "restored. Sources (original uploads) are in $D/sources -- point A= at that folder."
