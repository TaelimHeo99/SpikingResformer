# AGENTS.md

## Purpose
- This repository trains, evaluates, and profiles SpikingResformer models for ImageNet, CIFAR, and DVS datasets.
- The main entrypoint is `main.py`; most agent work will touch `models/`, `utils/`, and `configs/`.

## Repository Layout
- `main.py`: CLI entrypoint for training, evaluation, resume, transfer learning, and profiling.
- `configs/main/*.yaml`, `configs/direct_training/*.yaml`, `configs/transfer/*.yaml`: experiment presets.
- `models/spikingresformer.py`: model architecture and timm model registration.
- `models/submodules/layers.py`: spiking layers, wrappers, and custom ops.
- `utils/utils.py` and `utils/augment.py`: metrics, dataset wrappers, timers, profiling hooks, and augmentations.

## Dependencies And Environment
- The README documents install steps with `pip3 install torch torchvision` and `pip3 install tensorboard thop spikingjelly==0.0.0.0.14 cupy-cuda11x timm`.
- Python packaging metadata is not present (`pyproject.toml`, `setup.py`, `requirements*.txt`, and `Makefile` are absent).
- The code assumes CUDA is available for normal training and evaluation paths.
- Distributed training expects `torchrun` environment variables such as `RANK`, `WORLD_SIZE`, and `LOCAL_RANK`.
- Data directories are external and passed with `--data-path`; large datasets are intentionally gitignored.

## Canonical Commands

### Install
```bash
pip3 install torch torchvision
pip3 install tensorboard thop spikingjelly==0.0.0.0.14 cupy-cuda11x timm
```

### Train On ImageNet
```bash
torchrun --standalone --nnodes=1 --nproc-per-node=8 main.py \
  -c configs/main/spikingresformer_s.yaml \
  --data-path /path/to/dataset \
  --output-dir /path/to/output
```

### Evaluate A Checkpoint
```bash
python main.py \
  --model spikingresformer_s \
  --data-path /path/to/dataset \
  --resume /path/to/checkpoint.pth \
  --test-only
```

### Transfer Learning Example
```bash
python main.py \
  -c configs/transfer/cifar10.yaml \
  --data-path /path/to/dataset \
  --output-dir /path/to/output \
  --transfer /path/to/pretrained.pth
```

### Direct Training Example
```bash
python main.py \
  -c configs/direct_training/cifar10.yaml \
  --data-path /path/to/dataset \
  --output-dir /path/to/output
```

## Build, Lint, And Test Reality
- There is no dedicated build system in this repo.
- There is no configured linter or formatter in-repo; no `ruff`, `black`, `flake8`, `isort`, or `mypy` config files were found.
- There is no `tests/` directory and no unit-test suite was found.
- Validation is done through `python -m compileall`, `main.py --test-only`, or short smoke runs.

## Practical Verification Commands

### Fast Syntax Check
```bash
python -m compileall main.py models utils
```

### Single Evaluation Run (Closest Thing To A Single Test)
```bash
python main.py \
  -c configs/direct_training/cifar10.yaml \
  --data-path /path/to/dataset \
  --resume /path/to/checkpoint.pth \
  --test-only
```

### Single-Config Smoke Run
```bash
python main.py \
  -c configs/direct_training/cifar10.yaml \
  --data-path /path/to/dataset \
  --output-dir /tmp/spikingresformer-smoke \
  --epochs 1 \
  --batch-size 8 \
  --workers 0 \
  --print-freq 1
```

## Single Test Guidance
- Because the repo has no unit tests, there is no native `pytest path::test_name` workflow.
- For agent tasks, define a "single test" as one of:
  - `python -m compileall main.py models utils` for syntax-only changes.
  - One `python main.py ... --test-only` run against the most relevant config/checkpoint.
  - One 1-epoch smoke run with reduced batch size and workers for training-loop changes.
- If you add a real test suite in the future, prefer `pytest tests/test_file.py::test_name -q`, but that is not currently part of this repository.

## Existing Rule Files
- No `.cursor/rules/` directory was found.
- No `.cursorrules` file was found.
- No `.github/copilot-instructions.md` file was found.

## Style Baseline
- Match the current code style instead of introducing a new framework-wide style.
- Keep changes minimal and local; avoid broad refactors unless the task explicitly requires them.
- Preserve current CLI/config behavior unless the change is explicitly about altering it.

## Formatting
- Follow existing 4-space indentation.
- Keep imports and statements in a simple, readable vertical layout.
- Existing files use long lines in places; do not reflow unrelated code just to satisfy a formatter.
- Preserve short comment style; avoid adding comments for obvious code.

## Imports
- Follow the repository's loose import grouping style:
  - standard library first,
  - third-party libraries next,
  - local project imports last.
- Use relative imports inside `models/` subpackages when neighboring modules already do so.
- Avoid adding unused imports; the repo currently does not have an automated linter to catch them.

## Types
- Type hints are used selectively, not exhaustively.
- Add type hints when they clarify interfaces, especially for public helpers, tensor shapes, or optional scheduler/scaler arguments.
- Do not force full typing coverage across legacy code.
- When shape expectations matter, encode them in parameter names, docstrings, or error messages if full static typing is not practical.

## Naming Conventions
- Use `snake_case` for functions, variables, and module-level helpers.
- Use `PascalCase` for classes.
- Preserve domain-specific names already established in the codebase, such as `DSSA`, `GWFFN`, `TET`, `SOPMonitor`, and `DVSAugment`.
- Prefer descriptive names tied to model semantics (`input_size`, `num_classes`, `data_loader_test`) over generic placeholders.

## Error Handling
- Follow the existing approach of raising `ValueError` for unsupported modes, datasets, or layer names.
- Fail early on invalid tensor shapes or unsupported config values.
- Keep error messages concrete and parameter-specific.
- If you must catch broadly for checkpoint or IO workflows, log a warning with the consequence, as `main.py` already does near final checkpoint loading.

## Logging And Metrics
- Use the repo's logger setup instead of ad hoc `print` statements for runtime status.
- Keep log messages concise and training-oriented.
- Respect `is_main_process()` for side effects such as checkpoint writing and TensorBoard logging.

## Distributed And CUDA Assumptions
- Training code is written for CUDA-first execution.
- Be careful when adding CPU-only logic; many helpers allocate CUDA tensors directly.
- Do not break single-process execution when editing distributed sections.

## Config And CLI Conventions
- New training options should be exposed through `parse_args()` and be overridable by YAML config files.
- Keep YAML keys aligned with argparse destination names.
- When changing config behavior, update the relevant YAML examples under `configs/`.

## Model And Tensor Conventions
- The model accepts either `[B, C, H, W]` or `[B, T, C, H, W]` inputs and normalizes internally to `[T, B, C, H, W]`.
- `functional.reset_net(model)` is required after forward passes in training, evaluation, and testing flows.
- Preserve the current `step_mode='m'` assumptions in custom layers unless the task explicitly changes step semantics.

## What Agents Should Avoid
- Do not add a new tooling stack unless the task explicitly asks for it.
- Do not silently rename public CLI flags, config keys, model registry names, or checkpoint fields.
- Do not remove checkpoint compatibility without documenting a migration path.
- Do not commit generated logs, datasets, or checkpoints.

## Recommended Agent Workflow
- Read the relevant config file before changing training logic.
- Prefer the smallest possible verification command that matches the area changed.
- For architecture edits, inspect `models/spikingresformer.py` and `models/submodules/layers.py` together.
- For training-loop edits, inspect `main.py` and `utils/utils.py` together.
- If you cannot run a full experiment, state exactly what you validated and what remains unverified.
