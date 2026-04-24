python main.py \
  -c configs/transfer/dvs128gesture.yaml \
  --model spikingresformer_ti \
  --data-path ./datasets/DVS128Gesture \
  --output-dir ./logs/transfer_dvs128gesture_baseline \
  --transfer ./pretrained/spikingresformer_ti.pth \
  --wandb-project spikingresformer \
  --wandb-name transfer_dvs128gesture_baseline \
  --save-latest
