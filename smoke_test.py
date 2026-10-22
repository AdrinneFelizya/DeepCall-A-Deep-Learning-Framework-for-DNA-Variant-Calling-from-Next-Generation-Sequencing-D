"""Quick smoke test to verify the full pipeline works (synthetic, no pysam needed)."""
import os
import warnings
warnings.filterwarnings("ignore")
import torch
from torch.utils.data import DataLoader
import numpy as np

from config import *
from data.pileup_generator import PileupGenerator
from data.dataset import SyntheticPileupDataset
from models.cnn import VariantCNN
from models.resnet18 import VariantResNet18
from models.vgg16 import VariantVGG16
from training.trainer import Trainer
from training.metrics import compute_metrics, plot_confusion_matrix
from postprocessing.vcf_writer import VCFWriter

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Generating synthetic data...")
generator = PileupGenerator(
    height=IMAGE_HEIGHT, width=IMAGE_WIDTH, channels=IMAGE_CHANNELS
)
train_images, train_labels = generator.generate_dataset(2000, seed=42)
val_images, val_labels = generator.generate_dataset(400, seed=123)
test_images, test_labels = generator.generate_dataset(400, seed=456)

train_loader = DataLoader(
    SyntheticPileupDataset(train_images, train_labels),
    batch_size=64, shuffle=True,
)
val_loader = DataLoader(
    SyntheticPileupDataset(val_images, val_labels),
    batch_size=64, shuffle=False,
)
test_loader = DataLoader(
    SyntheticPileupDataset(test_images, test_labels),
    batch_size=64, shuffle=False,
)

for model_name, model, epochs in [
    ("CNN", VariantCNN(in_channels=IMAGE_CHANNELS, num_classes=NUM_CLASSES), 5),
    ("ResNet18", VariantResNet18(
        in_channels=IMAGE_CHANNELS, num_classes=NUM_CLASSES, pretrained=False
    ), 10),
    ("VGG16", VariantVGG16(
        in_channels=IMAGE_CHANNELS, num_classes=NUM_CLASSES, pretrained=False
    ), 5),
]:
    print(f"\n{'='*50}")
    print(f"Training {model_name}...")
    print(f"{'='*50}")

    model_lr = 1e-4 if model_name == "VGG16" else LEARNING_RATE
    trainer = Trainer(
        model=model, device=DEVICE, class_weights=CLASS_WEIGHTS,
        learning_rate=model_lr, weight_decay=WEIGHT_DECAY,
        model_name=model_name,
    )
    history = trainer.fit(
        train_loader=train_loader, val_loader=val_loader,
        epochs=epochs, save_dir=MODEL_DIR,
    )

    model.load_state_dict(
        torch.load(os.path.join(MODEL_DIR, f"{model_name}_best.pth"),
                   map_location=DEVICE)
    )
    predictions, labels, probs = trainer.predict(test_loader)
    metrics = compute_metrics(labels, predictions, num_classes=NUM_CLASSES)

    acc = metrics["accuracy"]
    print(f"\n{model_name} Test Accuracy: {acc:.4f}")
    for i, name in enumerate(CLASS_NAMES):
        p = metrics["precision"][i]
        r = metrics["recall"][i]
        f1 = metrics["f1"][i]
        print(f"  {name:12s} - P: {p:.4f} | R: {r:.4f} | F1: {f1:.4f}")

    vcf_writer = VCFWriter(
        os.path.join(OUTPUT_DIR, f"{model_name}_variants.vcf")
    )
    positions = np.arange(len(predictions)) * 100 + 1000
    vcf_writer.write(predictions, probs, positions=positions)

print(f"\n{'='*50}")
print("SMOKE TEST PASSED!")
print(f"{'='*50}")
print("All models trained, evaluated, and VCF generated successfully!")