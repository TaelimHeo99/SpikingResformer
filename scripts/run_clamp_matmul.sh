#!/bin/bash
# Clamp-matmul experiments (negative matmul outputs → 0)
# Usage: bash scripts/run_clamp_matmul.sh <data_root> <output_root>

DATA=${1:-./datasets}
OUT=${2:-./logs}

# Direct Training
python main.py -c configs/direct_training/cifar10.yaml \
    --data-path $DATA --output-dir $OUT/direct_cifar10_clamp --clamp-matmul --no-wandb

python main.py -c configs/direct_training/cifar100.yaml \
    --data-path $DATA --output-dir $OUT/direct_cifar100_clamp --clamp-matmul --no-wandb

python main.py -c configs/direct_training/cifar10dvs.yaml \
    --data-path $DATA/CIFAR10DVS --output-dir $OUT/direct_cifar10dvs_clamp --clamp-matmul --no-wandb

python main.py -c configs/direct_training/dvs128gesture.yaml \
    --data-path $DATA/DVS128Gesture --output-dir $OUT/direct_dvs128gesture_clamp --clamp-matmul --no-wandb

# Transfer Learning (ImageNet pretrained checkpoint required)
CKPT=${3:-./pretrained/spikingresformer_s.pth}

python main.py -c configs/transfer/cifar10.yaml \
    --data-path $DATA --output-dir $OUT/transfer_cifar10_clamp \
    --transfer $CKPT --clamp-matmul --no-wandb

python main.py -c configs/transfer/cifar100.yaml \
    --data-path $DATA --output-dir $OUT/transfer_cifar100_clamp \
    --transfer $CKPT --clamp-matmul --no-wandb

python main.py -c configs/transfer/cifar10dvs.yaml \
    --data-path $DATA/CIFAR10DVS --output-dir $OUT/transfer_cifar10dvs_clamp \
    --transfer $CKPT --clamp-matmul --no-wandb

python main.py -c configs/transfer/dvs128gesture.yaml \
    --data-path $DATA/DVS128Gesture --output-dir $OUT/transfer_dvs128gesture_clamp \
    --transfer $CKPT --clamp-matmul --no-wandb
