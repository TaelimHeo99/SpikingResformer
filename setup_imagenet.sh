#!/bin/bash
# ImageNet 압축 풀기 및 폴더 구조 정리
# Usage: bash setup_imagenet.sh <tar_dir> <output_dir>
#
# <tar_dir> 에 아래 파일이 있어야 함:
#   ILSVRC2012_img_train.tar (~138GB)
#   ILSVRC2012_img_val.tar   (~6.3GB)
#
# 결과 구조:
#   <output_dir>/train/n01440764/*.JPEG ...
#   <output_dir>/val/n01440764/*.JPEG   ...

TAR_DIR=${1:?Usage: $0 <tar_dir> <output_dir>}
OUT=${2:?Usage: $0 <tar_dir> <output_dir>}

mkdir -p $OUT/train $OUT/val

echo "[1/3] Extracting train set (시간 많이 걸림)..."
tar -xf $TAR_DIR/ILSVRC2012_img_train.tar -C $OUT/train
# 각 클래스별 tar 풀기
for f in $OUT/train/*.tar; do
    dir=$OUT/train/$(basename $f .tar)
    mkdir -p $dir
    tar -xf $f -C $dir
    rm $f
done

echo "[2/3] Extracting val set..."
tar -xf $TAR_DIR/ILSVRC2012_img_val.tar -C $OUT/val

echo "[3/3] Organizing val set by class..."
# valprep.sh 방식: devkit의 val 레이블로 폴더 정리
wget -qO /tmp/valprep.sh https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh
cd $OUT/val && bash /tmp/valprep.sh
cd -

echo "Done. Structure:"
ls $OUT/train | head -5
ls $OUT/val   | head -5
