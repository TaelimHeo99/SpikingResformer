python main.py \
  -c configs/transfer/dvs128gesture.yaml \
  --model spikingresformer_ti \
  --data-path ./datasets/DVS128Gesture \
  --output-dir ./logs/transfer_dvs128gesture_clamp \
  --transfer ./pretrained/spikingresformer_ti.pth \
  --clamp-matmul \
  --wandb-project spikingresformer \
  --wandb-name transfer_dvs128gesture_clamp \
  --save-latest
