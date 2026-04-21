python main.py \
  -c configs/transfer/dvs128gesture.yaml \
  --model spikingresformer_ti \
  --data-path ./datasets/DVS128Gesture \
  --output-dir ./logs/transfer_dvs128gesture_baseline_test \
  --resume ./logs/transfer_dvs128gesture_baseline/checkpoint_max_acc1.pth \
  --test-only \
  --no-wandb
