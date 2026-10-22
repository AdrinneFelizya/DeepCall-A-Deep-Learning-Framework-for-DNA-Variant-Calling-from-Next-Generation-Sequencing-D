"""
Prediction engine for DeepCall backend.
Loads trained models and predicts variants from pileup images.
"""
from __future__ import annotations

import os
import numpy as np
import torch
import yaml
from typing import Dict


class ModelPredictor:
    """Loads a trained model and makes predictions on pileup images."""

    def __init__(self, config_path: str = "config.yaml", model_path: str = None):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = None
        self.model_name = None
        self.class_names = ["No Variant", "SNP", "Insertion", "Deletion"]

        if model_path:
            self.load_model(model_path)

    @staticmethod
    def _infer_architecture(model_path: str) -> str:
        name = os.path.basename(model_path).lower()
        for arch in ("resnet18", "transformer", "vgg16", "cnn"):
            if arch in name:
                return arch
        return "cnn"

    def load_model(self, model_path: str, architecture: str = None):
        """Load a trained model from checkpoint."""
        if architecture is None:
            architecture = self._infer_architecture(model_path)
        self.model_name = architecture
        num_classes = self.config["model"].get("num_classes_synthetic", 4)
        in_channels = self.config["data"]["synth_channels"]

        if architecture == "cnn":
            from models.cnn import VariantCNN
            self.model = VariantCNN(
                in_channels=in_channels, num_classes=num_classes
            )
        elif architecture == "resnet18":
            from models.resnet18 import VariantResNet18
            self.model = VariantResNet18(
                in_channels=in_channels, num_classes=num_classes, pretrained=False
            )
        elif architecture == "vgg16":
            from models.vgg16 import VariantVGG16
            self.model = VariantVGG16(
                in_channels=in_channels, num_classes=num_classes, pretrained=False
            )
        elif architecture == "transformer":
            from models.transformer_caller import VariantCallerTransformer
            t = self.config["model"]["transformer"]
            self.model = VariantCallerTransformer(
                in_channels=in_channels, num_classes=num_classes,
                d_model=t["d_model"], nhead=t["nhead"],
                num_layers=t["num_layers"],
                dim_feedforward=t["dim_feedforward"],
                dropout=t["dropout"],
            )
        else:
            raise ValueError(f"Unknown architecture: {architecture}")

        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device)
        )
        self.model.to(self.device)
        self.model.eval()
        print(f"Loaded {architecture} model from {model_path}")

    def predict(self, image: np.ndarray) -> Dict:
        """Predict variant class from a single pileup image.

        Parameters
        ----------
        image : np.ndarray
            Shape (channels, height, width), values in [0, 1].

        Returns
        -------
        dict with keys: class, class_name, confidence, probabilities
        """
        if self.model is None:
            raise ValueError("No model loaded. Call load_model() first.")

        x = torch.from_numpy(image).unsqueeze(0).float().to(self.device)

        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        pred_idx = int(probs.argmax())
        confidence = float(probs[pred_idx])

        return {
            "class": pred_idx,
            "class_name": self.class_names[pred_idx],
            "confidence": confidence,
            "probabilities": {
                self.class_names[i]: float(p) for i, p in enumerate(probs)
            },
        }

    def predict_batch(self, images: np.ndarray) -> list:
        """Predict on a batch of images."""
        return [self.predict(img) for img in images]


__all__ = ["ModelPredictor"]