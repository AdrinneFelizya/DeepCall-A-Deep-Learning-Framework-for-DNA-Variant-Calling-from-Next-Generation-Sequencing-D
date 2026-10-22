"""
Training pipeline exposed for the frontend.
Trains CNN/ResNet18/VGG16 on synthetic pileup data.
"""
from __future__ import annotations

import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml
from typing import Dict

from data.pileup_generator import PileupGenerator
from data.dataset import SyntheticPileupDataset
from models.cnn import VariantCNN
from models.resnet18 import VariantResNet18
from models.vgg16 import VariantVGG16
from models.transformer_caller import VariantCallerTransformer
from training.trainer import Trainer
from training.metrics import compute_metrics, plot_confusion_matrix


def train_model(config_path: str = "config.yaml",
                architecture: str = "cnn",
                epochs: int = 15,
                save_dir: str = "saved_models") -> Dict:
    """Train a model and return results."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = config["data"]

    # Generate synthetic data
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

    train_loader = DataLoader(
        SyntheticPileupDataset(train_images, train_labels),
        batch_size=config["train"]["batch_size"], shuffle=True,
    )
    val_loader = DataLoader(
        SyntheticPileupDataset(val_images, val_labels),
        batch_size=config["train"]["batch_size"], shuffle=False,
    )
    test_loader = DataLoader(
        SyntheticPileupDataset(test_images, test_labels),
        batch_size=config["train"]["batch_size"], shuffle=False,
    )

    num_classes = config["model"].get("num_classes_synthetic", 4)
    class_names = ["No Variant", "SNP", "Insertion", "Deletion"]

    # Build model
    if architecture == "cnn":
        model = VariantCNN(in_channels=d["synth_channels"], num_classes=num_classes)
    elif architecture == "resnet18":
        model = VariantResNet18(
            in_channels=d["synth_channels"], num_classes=num_classes, pretrained=False
        )
    elif architecture == "vgg16":
        model = VariantVGG16(
            in_channels=d["synth_channels"], num_classes=num_classes, pretrained=False
        )
    elif architecture == "transformer":
        t = config["model"]["transformer"]
        model = VariantCallerTransformer(
            in_channels=d["synth_channels"], num_classes=num_classes,
            d_model=t["d_model"], nhead=t["nhead"],
            num_layers=t["num_layers"],
            dim_feedforward=t["dim_feedforward"],
            dropout=t["dropout"],
        )
    else:
        raise ValueError(f"Unknown architecture: {architecture}")

    model = model.to(device)

    lr = float(config["train"]["lr"])
    if "vgg" in architecture.lower():
        lr = min(lr, 1e-4)

    trainer = Trainer(
        model=model, device=device,
        class_weights=config["train"].get("class_weights", [1.0, 5.0, 10.0, 10.0]),
        learning_rate=lr,
        weight_decay=float(config["train"]["weight_decay"]),
        model_name=architecture,
        use_focal_loss=config["train"]["focal_loss"],
        focal_gamma=float(config["train"]["focal_gamma"]),
        label_smoothing=float(config["train"].get("label_smoothing", 0.0)),
    )

    os.makedirs(save_dir, exist_ok=True)
    history = trainer.fit(train_loader, val_loader, epochs=epochs, save_dir=save_dir)

    # Evaluate
    best_path = os.path.join(save_dir, f"{architecture}_best.pth")
    model.load_state_dict(torch.load(best_path, map_location=device))
    predictions, labels, probs = trainer.predict(test_loader)
    metrics = compute_metrics(labels, predictions, num_classes=num_classes)

    # Save confusion matrix
    os.makedirs(config.get("output_dir", "output"), exist_ok=True)
    cm_path = os.path.join(config["output_dir"], f"{architecture}_confusion_matrix.png")
    plot_confusion_matrix(metrics["confusion_matrix"], class_names, save_path=cm_path)

    return {
        "architecture": architecture,
        "accuracy": float(metrics["accuracy"]),
        "precision": metrics["precision"].tolist(),
        "recall": metrics["recall"].tolist(),
        "f1": metrics["f1"].tolist(),
        "confusion_matrix": metrics["confusion_matrix"].tolist(),
        "class_names": class_names,
        "history": history,
        "model_path": best_path,
        "cm_path": cm_path,
    }


__all__ = ["train_model"]