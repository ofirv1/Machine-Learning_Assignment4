from __future__ import annotations

import argparse
from pathlib import Path
import json

import torch

from src.data import (
    ensure_oxford102,
    make_random_split,
    split_summary,
    build_transforms,
    build_datasets_for_split,
    build_dataloaders,
)
from src.models import build_vgg19_model, YOLOv5Classifier, unfreeze_last_yolo_layers
from src.train_utils import train_model, merge_histories


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str, default=".", help="Project root for data/ directory (default: current dir)")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed1", type=int, default=123)
    p.add_argument("--seed2", type=int, default=456)

    # VGG
    p.add_argument("--epochs-vgg", type=int, default=10)
    p.add_argument("--lr-vgg", type=float, default=1e-3)

    # YOLO stages
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--epochs-yolo-s1", type=int, default=2)
    p.add_argument("--lr-yolo-s1", type=float, default=1e-3)
    p.add_argument("--epochs-yolo-s2", type=int, default=8)
    p.add_argument("--lr-yolo-s2", type=float, default=1e-4)
    p.add_argument("--yolo-unfreeze-last", type=int, default=30)

    p.add_argument("--dry-run", action="store_true", help="Run only a single batch sanity-check and exit")
    p.add_argument("--out-dir", type=str, default="results", help="Where to save histories json")
    args = p.parse_args()

    root = Path(args.root).resolve()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = (device.type == "cuda")

    print(f"Device: {device}")

    image_paths, labels = ensure_oxford102(root)
    num_classes = 102

    split1 = make_random_split(labels, args.seed1)
    split2 = make_random_split(labels, args.seed2)
    split_summary("SPLIT1", split1, labels)
    split_summary("SPLIT2", split2, labels)

    # transforms
    tf_train = build_transforms(args.img_size, train=True)
    tf_eval = build_transforms(args.img_size, train=False)

    out_dir = Path(args.out_dir)

    def run_one_split(split_name: str, split):
        # datasets/dataloaders
        ds_train, ds_val, ds_test = build_datasets_for_split(image_paths, labels, split, tf_train, tf_eval)
        dl_train, dl_val, dl_test = build_dataloaders(
            ds_train, ds_val, ds_test,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=pin_memory
        )

        # DRY RUN: one batch forward/backward sanity
        if args.dry_run:
            x, y = next(iter(dl_train))
            x = x.to(device)
            y = y.to(device)

            # VGG sanity
            vgg = build_vgg19_model(num_classes).to(device)
            logits = vgg(x)
            print(f"[DRY] VGG logits: {logits.shape}")

            # YOLO sanity
            yolo = YOLOv5Classifier(num_classes=num_classes, yolo_variant="yolov5s", pretrained=True, freeze_backbone=True).to(device)
            yolo.init_head(args.img_size, device)
            logits2 = yolo(x)
            print(f"[DRY] YOLO logits: {logits2.shape}")

            return

        # ----------------------------
        # VGG
        # ----------------------------
        print(f"\n===== VGG19 TRAINING – {split_name} =====\n")
        vgg = build_vgg19_model(num_classes).to(device)
        hist_vgg = train_model(vgg, dl_train, dl_val, dl_test, epochs=args.epochs_vgg, lr=args.lr_vgg, device=device)
        save_json(hist_vgg, out_dir / f"vgg19_{split_name.lower()}.json")

        # ----------------------------
        # YOLO - stage 1 (head only)
        # ----------------------------
        print(f"\n===== YOLOv5 TRAINING – {split_name} (stage1) =====\n")
        yolo = YOLOv5Classifier(num_classes=num_classes, yolo_variant="yolov5s", pretrained=True, freeze_backbone=True).to(device)
        yolo.init_head(args.img_size, device)
        hist_yolo_s1 = train_model(yolo, dl_train, dl_val, dl_test, epochs=args.epochs_yolo_s1, lr=args.lr_yolo_s1, device=device)
        save_json(hist_yolo_s1, out_dir / f"yolov5_{split_name.lower()}_stage1.json")

        # ----------------------------
        # YOLO - stage 2 (unfreeze last blocks)
        # ----------------------------
        print(f"\n===== YOLOv5 TRAINING – {split_name} (stage2) =====\n")
        unfreeze_last_yolo_layers(yolo, n_last=args.yolo_unfreeze_last)
        hist_yolo_s2 = train_model(yolo, dl_train, dl_val, dl_test, epochs=args.epochs_yolo_s2, lr=args.lr_yolo_s2, device=device)
        save_json(hist_yolo_s2, out_dir / f"yolov5_{split_name.lower()}_stage2.json")

        hist_yolo = merge_histories(hist_yolo_s1, hist_yolo_s2)
        save_json(hist_yolo, out_dir / f"yolov5_{split_name.lower()}.json")

        # summary line
        print("\n--- FINAL EPOCH TEST ACC ---")
        print(f"VGG19  {split_name}: {hist_vgg['test_acc'][-1]:.4f}")
        print(f"YOLOv5 {split_name}: {hist_yolo['test_acc'][-1]:.4f}")

    run_one_split("Split1", split1)
    run_one_split("Split2", split2)


if __name__ == "__main__":
    main()
