"""
Training Pipeline with Early Stopping, LR Scheduling, and Class Weighting.
Enhanced to support both synthetic (DL.zip) and real (BAM) datasets.
"""
from __future__ import annotations

import os
import json
import torch
import torch.nn as nn
import numpy as np
from datetime import datetime


class Trainer:
    """Trainer class for variant calling models."""

    def __init__(
        self,
        model,
        device,
        class_weights=None,
        learning_rate=1e-3,
        weight_decay=1e-4,
        model_name="model",
        use_focal_loss=False,
        focal_gamma=2.0,
        label_smoothing=0.0,
    ):
        self.model = model.to(device)
        self.device = device
        self.model_name = model_name

        if class_weights is not None:
            class_weights = torch.FloatTensor(class_weights).to(device)

        if use_focal_loss:
            from training.focal_loss import FocalLoss
            self.criterion = FocalLoss(
                gamma=focal_gamma,
                weight=class_weights,
                label_smoothing=label_smoothing,
            )
        else:
            self.criterion = nn.CrossEntropyLoss(
                weight=class_weights, label_smoothing=label_smoothing
            )

        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay)
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5
        )

        self.best_val_loss = float("inf")
        self.patience = 10
        self.counter = 0

        self.history = {
            "train_loss": [], "train_acc": [],
            "val_loss": [], "val_acc": [],
        }

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in dataloader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        return total_loss / total, correct / total

    def validate(self, dataloader):
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return total_loss / total, correct / total

    def fit(self, train_loader, val_loader, epochs=50, save_dir="./saved_models"):
        os.makedirs(save_dir, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Training {self.model_name}")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Epochs: {epochs}")
        print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']}")
        print(f"Weight Decay: {self.optimizer.param_groups[0]['weight_decay']}")
        print(f"{'='*60}\n")

        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate(val_loader)
            self.scheduler.step(val_loss)

            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)

            print(
                f"Epoch [{epoch+1}/{epochs}] "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
            )

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.counter = 0
                save_path = os.path.join(save_dir, f"{self.model_name}_best.pth")
                torch.save(self.model.state_dict(), save_path)
                print(f"  -> Best model saved (val_loss: {val_loss:.4f})")
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    print(f"\nEarly stopping triggered after {epoch+1} epochs")
                    break

        history_path = os.path.join(save_dir, f"{self.model_name}_history.json")
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)

        print(f"\nTraining complete! Best val loss: {self.best_val_loss:.4f}")
        return self.history

    def predict(self, dataloader):
        self.model.eval()
        all_predictions = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(self.device)
                outputs = self.model(images)
                probs = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)

                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_probs.extend(probs.cpu().numpy())

        return (
            np.array(all_predictions),
            np.array(all_labels),
            np.array(all_probs),
        )


__all__ = ["Trainer"]