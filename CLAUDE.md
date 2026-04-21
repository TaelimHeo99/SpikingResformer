# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Repo Does

SpikingResformer is a Spiking Neural Network (SNN) research implementation (CVPR 2024) that trains and evaluates models on ImageNet, CIFAR-10/100, and DVS-based datasets. There is no build system, no linter config, and no test suite.

## Commands

### Install
```bash
pip3 install torch torchvision
pip3 install tensorboard thop spikingjelly==0.0.0.0.14 cupy-cuda11x timm
```

### Train on ImageNet (8 GPUs)
```bash
torchrun --standalone --nnodes=1 --nproc-per-node=8 main.py \
  -c configs/main/spikingresformer_s.yaml \
  --data-path /path/to/dataset \
  --output-dir /path/to/output
```

### Evaluate a checkpoint
```bash
python main.py \
  --model spikingresformer_s \
  --data-path /path/to/dataset \
  --resume /path/to/checkpoint.pth \
  --test-only
```

### Transfer learning (CIFAR / DVS)
```bash
python main.py \
  -c configs/transfer/cifar10.yaml \
  --data-path /path/to/dataset \
  --output-dir /path/to/output \
  --transfer /path/to/pretrained.pth
```

### Syntax check (no dataset needed)
```bash
python -m compileall main.py models utils
```

### Smoke run (1 epoch, no checkpoint needed)
```bash
python main.py \
  -c configs/direct_training/cifar10.yaml \
  --data-path /path/to/dataset \
  --output-dir /tmp/smoke \
  --epochs 1 --batch-size 8 --workers 0 --print-freq 1
```

## Architecture Overview

### Entry Point
`main.py` handles everything: argument parsing (YAML config + CLI overrides), dataset loading, model creation, training loop, evaluation, checkpointing, and profiling. `--test-only` runs `test()` which also reports MACs, params, and spike operations (SOPs) via `SOPMonitor`.

### Model (`models/spikingresformer.py`)
Four size variants registered via timm's `@register_model`: `spikingresformer_{ti,s,m,l}` for ImageNet, plus `spikingresformer_cifar` and `spikingresformer_dvsg` for smaller datasets. All are instances of `SpikingResformer`, which stacks:
- **Prologue**: Conv7×7 stride-2 → BN → MaxPool (7×7 for ImageNet, 3×3 for CIFAR)
- **Stages**: alternating `DSSA` (Dynamic Spike-based Self-Attention) and `GWFFN` (Group-wise Feedforward Network) blocks, with `DownsampleLayer` (stride-2 Conv3×3) between stages
- **Head**: AdaptiveAvgPool2d → Linear classifier

Output shape is `[T, B, num_classes]`; metrics use `output.mean(0)` (temporal average).

### Spiking Layers (`models/submodules/layers.py`)
Wrappers for spikingjelly neurons (`IF`, `LIF`, `PLIF`) and custom ops (`Conv3x3`, `Conv1x1`, `Linear`, `SpikingMatmul`). The `BN` wrapper reshapes `[T,B,C,H,W]` → `[B,C,H,W]` for BatchNorm, then restores the temporal dimension.

### Temporal Dimension Contract
- The model normalizes any `[B,C,H,W]` input to `[T,B,C,H,W]` internally (T=4 by default).
- **`functional.reset_net(model)` must be called after every forward pass** (resets spikingjelly neuron states). This is already done in `train_one_epoch`, `evaluate`, and `test`.

### Config System
YAML files under `configs/{main,transfer,direct_training}/` set all hyperparameters. Keys match argparse destinations exactly. CLI flags override YAML values. New options must go through `parse_args()`.

### Distributed Training
Uses `torchrun` + `torch.distributed`. All checkpoint writes and TensorBoard/W&B logging are guarded by `is_main_process()`. `--test-only` must be run on a single process.

## Key Invariants

- Never rename public CLI flags, config keys, model registry names, or checkpoint fields without a migration path.
- When editing the training loop, read `main.py` and `utils/utils.py` together. When editing the architecture, read `models/spikingresformer.py` and `models/submodules/layers.py` together.
- `PLIF` neurons have learnable tau via `.w`; exclude `.w` from weight decay (already handled in optimizer setup).
- Do not commit logs, checkpoints, or datasets (`logs/`, `datasets/` are gitignored).
