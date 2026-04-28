#!/bin/bash
DATA=${1:-./datasets/imagenet}

python main.py \
  -c configs/main/spikingresformer_ti.yaml \
  --data-path $DATA \
  --output-dir ./logs/imagenet_ti_clamp \
  --batch-size 256 \
  --clamp-matmul clamp \
  --wandb-project spikingresformer \
  --wandb-name imagenet_ti_clamp \
  --save-latest
