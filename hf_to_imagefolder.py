"""HuggingFace imagenet-1k (parquet) → ImageFolder 구조로 변환"""
import os
import glob
import argparse
from datasets import load_dataset

parser = argparse.ArgumentParser()
parser.add_argument("--input-dir", default="/data/imagenet")
parser.add_argument("--output-dir", default="./datasets/imagenet")
parser.add_argument("--split", default="all", choices=["all", "train", "validation"])
args = parser.parse_args()

splits = ["train", "validation"] if args.split == "all" else [args.split]
split_map = {"train": "train", "validation": "val"}

for split in splits:
    print(f"Loading {split}...")
    parquet_files = sorted(glob.glob(os.path.join(args.input_dir, "data", f"{split}-*.parquet")))
    ds = load_dataset("parquet", data_files={split: parquet_files}, split=split)
    label_names = ds.features["label"].names
    out_dir = os.path.join(args.output_dir, split_map[split])

    for i, example in enumerate(ds):
        class_name = label_names[example["label"]]
        class_dir = os.path.join(out_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        img_path = os.path.join(class_dir, f"{i:08d}.JPEG")
        example["image"].save(img_path, "JPEG")

        if i % 10000 == 0:
            print(f"  {split}: {i}/{len(ds)}")

    print(f"{split} done → {out_dir}")
