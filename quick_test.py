"""Quick lightweight test: verify imports, CNN training, evaluation, VCF output."""
import os
import torch
from torch.utils.data import DataLoader
import numpy as np

from config import *
from data.pileup_generator import PileupGenerator
from data.dataset import SyntheticPileupDataset
from models.cnn import VariantCNN
from training.trainer import Trainer
from training.metrics import compute_metrics, plot_confusion_matrix
from postprocessing.vcf_writer import VCFWriter

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Device:", DEVICE)
print("Generating synthetic data...")
generator = PileupGenerator(height=IMAGE_HEIGHT, width=IMAGE_WIDTH, channels=IMAGE_CHANNELS)
train_images, train_labels = generator.generate_dataset(800, seed=42)
val_images, val_labels = generator.generate_dataset(200, seed=123)
test_images, test_labels = generator.generate_dataset(200, seed=456)

train_loader = DataLoader(SyntheticPileupDataset(train_images, train_labels), batch_size=32, shuffle=True)
val_loader = DataLoader(SyntheticPileupDataset(val_images, val_labels), batch_size=32, shuffle=False)
test_loader = DataLoader(SyntheticPileupDataset(test_images, test_labels), batch_size=32, shuffle=False)

print("Training CNN (3 epochs)...")
model = VariantCNN(in_channels=IMAGE_CHANNELS, num_classes=NUM_CLASSES)
trainer = Trainer(model=model, device=DEVICE, class_weights=CLASS_WEIGHTS,
                  learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY, model_name="CNN")
trainer.fit(train_loader, val_loader, epochs=3, save_dir=MODEL_DIR)

model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "CNN_best.pth"), map_location=DEVICE))
predictions, labels, probs = trainer.predict(test_loader)
metrics = compute_metrics(labels, predictions, num_classes=NUM_CLASSES)

print(f"\nCNN Test Accuracy: {metrics['accuracy']:.4f}")
for i, name in enumerate(CLASS_NAMES):
    print(f"  {name:12s} - P: {metrics['precision'][i]:.4f} | "
          f"R: {metrics['recall'][i]:.4f} | F1: {metrics['f1'][i]:.4f}")

# Save confusion matrix
cm_path = os.path.join(OUTPUT_DIR, "CNN_confusion_matrix.png")
plot_confusion_matrix(metrics['confusion_matrix'], CLASS_NAMES, save_path=cm_path)

# Write VCF
vcf_path = os.path.join(OUTPUT_DIR, "CNN_variants.vcf")
vcf_writer = VCFWriter(vcf_path)
positions = np.arange(len(predictions)) * 100 + 1000
vcf_writer.write(predictions, probs, positions=positions)

print(f"\nQuick test PASSED!")
print(f"  Model saved: {MODEL_DIR}/CNN_best.pth")
print(f"  Confusion matrix: {cm_path}")
print(f"  VCF output: {vcf_path}")