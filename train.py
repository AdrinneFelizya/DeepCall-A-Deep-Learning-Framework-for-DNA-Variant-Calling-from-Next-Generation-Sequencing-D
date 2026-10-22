"""
Main Training Script for DNA Variant Calling.

Supports two modes:
  * synthetic: trains CNN/ResNet18/VGG16 on synthetic pileups (from DL.zip,
    no pysam needed) — good for smoke testing and paper baselines.
  * real: trains on real BAM pileups with chromosome-wise split — for real-life
    application and conference paper results on GIAB/1000 Genomes data.

Usage:
    python train.py --mode synthetic --config config.yaml
    python train.py --mode real --config config.yaml
"""

from __future__ import annotations

import argparse
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

from config import *  # noqa: F401  (synthetic-mode constants)
from data.pileup_generator import PileupGenerator
from data.dataset import (
    SyntheticPileupDataset,
    RealPileupDataset,
    make_real_dataloaders,
)
from models.cnn import VariantCNN
from models.resnet18 import VariantResNet18
from models.vgg16 import VariantVGG16
from models.transformer_caller import VariantCallerTransformer
from training.trainer import Trainer
from training.metrics import compute_metrics, plot_confusion_matrix


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_model(config: dict, device: torch.device, num_classes: int,
                in_channels_override: int | None = None):
    m = config["model"]
    arch = m["architecture"]
    in_ch = in_channels_override if in_channels_override is not None else m["in_channels"]

    if arch == "cnn":
        return VariantCNN(
            in_channels=in_ch, num_classes=num_classes,
            dropout_rate=m.get("dropout", 0.5),
        ).to(device)
    if arch == "resnet18":
        return VariantResNet18(
            in_channels=in_ch, num_classes=num_classes,
            pretrained=m.get("pretrained", False),
            dropout_rate=m.get("dropout", 0.5),
        ).to(device)
    if arch == "vgg16":
        return VariantVGG16(
            in_channels=in_ch, num_classes=num_classes,
            pretrained=m.get("pretrained", False),
            dropout_rate=m.get("dropout", 0.5),
        ).to(device)
    if arch == "transformer":
        t = m["transformer"]
        return VariantCallerTransformer(
            in_channels=in_ch, num_classes=num_classes,
            d_model=t["d_model"], nhead=t["nhead"],
            num_layers=t["num_layers"],
            dim_feedforward=t["dim_feedforward"],
            dropout=t["dropout"],
        ).to(device)
    raise ValueError(f"Unknown architecture: {arch}")


def train_synthetic(config: dict, device: torch.device):
    """Train on synthetic pileups (DL.zip mode)."""
    print("\n" + "=" * 60)
    print("DNA Variant Calling - Training (SYNTHETIC mode)")
    print("=" * 60)

    os.makedirs(config["train"].get("checkpoint_dir", "saved_models"), exist_ok=True)
    os.makedirs(config.get("output_dir", "output"), exist_ok=True)

    # Synthetic data constants from config
    d = config["data"]
    generator = PileupGenerator(
        height=d["synth_height"], width=d["synth_width"],
        channels=d["synth_channels"],
    )

    train_images, train_labels = generator.generate_dataset(
        d["num_train_samples"], seed=42
    )
    val_images, val_labels = generator.generate_dataset(
        d["num_val_samples"], seed=123
    )
    test_images, test_labels = generator.generate_dataset(
        d["num_test_samples"], seed=456
    )

    print(f"  Train: {len(train_images)} samples")
    print(f"  Val:   {len(val_images)} samples")
    print(f"  Test:  {len(test_images)} samples")

    num_classes = config["model"].get("num_classes_synthetic", 4)
    class_names = ["No Variant", "SNP", "Insertion", "Deletion"]

    train_loader = DataLoader(
        SyntheticPileupDataset(train_images, train_labels),
        batch_size=config["train"]["batch_size"], shuffle=True, num_workers=0,
    )
    val_loader = DataLoader(
        SyntheticPileupDataset(val_images, val_labels),
        batch_size=config["train"]["batch_size"], shuffle=False, num_workers=0,
    )
    test_loader = DataLoader(
        SyntheticPileupDataset(test_images, test_labels),
        batch_size=config["train"]["batch_size"], shuffle=False, num_workers=0,
    )

    arch = config["model"]["architecture"]
    model = build_model(config, device, num_classes,
                        in_channels_override=d["synth_channels"])

    # VGG16 is sensitive to the default learning rate; scale it down.
    lr = float(config["train"]["lr"])
    if "vgg" in arch.lower():
        lr = min(lr, 1e-4)

    trainer = Trainer(
        model=model, device=device,
        class_weights=CLASS_WEIGHTS,
        learning_rate=lr,
        weight_decay=float(config["train"]["weight_decay"]),
        model_name=arch,
        use_focal_loss=config["train"]["focal_loss"],
        focal_gamma=float(config["train"]["focal_gamma"]),
        label_smoothing=float(config["train"].get("label_smoothing", 0.0)),
    )

    epochs = config["train"].get("synthetic_epochs", 15)
    history = trainer.fit(
        train_loader, val_loader,
        epochs=epochs,
        save_dir=config["train"]["checkpoint_dir"],
    )

    # Load best and evaluate
    best_path = os.path.join(
        config["train"]["checkpoint_dir"], f"{arch}_best.pth"
    )
    model.load_state_dict(torch.load(best_path, map_location=device))
    predictions, labels, probs = trainer.predict(test_loader)
    metrics = compute_metrics(labels, predictions, num_classes=num_classes)

    print(f"\n--- {arch} Test Results ---")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    for i, name in enumerate(class_names):
        print(
            f"  {name:12s} - P: {metrics['precision'][i]:.4f} | "
            f"R: {metrics['recall'][i]:.4f} | F1: {metrics['f1'][i]:.4f}"
        )

    cm_path = os.path.join(config.get("output_dir", "output"),
                           f"{arch}_confusion_matrix.png")
    plot_confusion_matrix(
        metrics["confusion_matrix"], class_names, save_path=cm_path
    )
    print(f"Confusion matrix saved to: {cm_path}")

    print("\nSynthetic training complete!")


def train_real(config: dict, device: torch.device):
    """Train on real BAM pileups with chromosome-wise split."""
    print("\n" + "=" * 60)
    print("DNA Variant Calling - Training (REAL mode)")
    print("=" * 60)

    train_loader, val_loader, test_loader = make_real_dataloaders(
        bam_path=config["data"]["bam_file"],
        config=config,
        truth_vcf_path=config["data"]["truth_vcf"],
    )

    num_classes = config["model"]["num_classes"]  # 3: hom-ref/het/hom-alt
    model = build_model(config, device, num_classes)

    trainer = Trainer(
        model=model, device=device,
        learning_rate=float(config["train"]["lr"]),
        weight_decay=float(config["train"]["weight_decay"]),
        model_name=config["model"]["architecture"],
        use_focal_loss=config["train"]["focal_loss"],
        focal_gamma=float(config["train"]["focal_gamma"]),
        label_smoothing=float(config["train"].get("label_smoothing", 0.0)),
    )

    epochs = config["train"]["epochs"]
    history = trainer.fit(
        train_loader, val_loader,
        epochs=epochs,
        save_dir=config["train"]["checkpoint_dir"],
    )

    print("\nReal-data training complete!")
    print(f"Best checkpoint: {config['train']['checkpoint_dir']}/"
          f"{config['model']['architecture']}_best.pth")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="synthetic",
                        choices=["synthetic", "real"])
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)

    if config["train"]["device"] == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(config["train"]["device"])

    print(f"Using device: {device}")
    print(f"Mode: {args.mode}")

    if args.mode == "synthetic":
        train_synthetic(config, device)
    else:
        train_real(config, device)


if __name__ == "__main__":
    main()