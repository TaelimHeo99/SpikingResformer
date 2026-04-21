#!/bin/bash
# Baseline experiments (original paper code)
# Usage: bash scripts/run_baseline.sh <data_root> <output_root>

DATA=${1:-./datasets}
OUT=${2:-./logs}

# Direct Training
python main.py -c configs/direct_training/cifar10.yaml \
    --data-path $DATA --output-dir $OUT/direct_cifar10_baseline --no-wandb

python main.py -c configs/direct_training/cifar100.yaml \
    --data-path $DATA --output-dir $OUT/direct_cifar100_baseline --no-wandb

python main.py -c configs/direct_training/cifar10dvs.yaml \
    --data-path $DATA/CIFAR10DVS --output-dir $OUT/direct_cifar10dvs_baseline --no-wandb

python main.py -c configs/direct_training/dvs128gesture.yaml \
    --data-path $DATA/DVS128Gesture --output-dir $OUT/direct_dvs128gesture_baseline --no-wandb

# Transfer Learning (ImageNet pretrained checkpoint required)
# Download from: https://drive.google.com/drive/folders/1sAFHKF9QZC2P0aSoC9bdLqgW8TfLjFSC
CKPT=${3:-./pretrained/spikingresformer_s.pth}

python main.py -c configs/transfer/cifar10.yaml \
    --data-path $DATA --output-dir $OUT/transfer_cifar10_baseline \
    --transfer $CKPT --no-wandb

python main.py -c configs/transfer/cifar100.yaml \
    --data-path $DATA --output-dir $OUT/transfer_cifar100_baseline \
    --transfer $CKPT --no-wandb

python main.py -c configs/transfer/cifar10dvs.yaml \
    --data-path $DATA/CIFAR10DVS --output-dir $OUT/transfer_cifar10dvs_baseline \
    --transfer $CKPT --no-wandb

python main.py -c configs/transfer/dvs128gesture.yaml \
    --data-path $DATA/DVS128Gesture --output-dir $OUT/transfer_dvs128gesture_baseline \
    --transfer $CKPT --no-wandb
