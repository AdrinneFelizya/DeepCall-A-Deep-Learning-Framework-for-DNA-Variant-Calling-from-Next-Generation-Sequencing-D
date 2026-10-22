"""
Generate sample pileup images for testing the DeepCall app.
Run this to create .npy files you can upload in the Predict tab.
"""
from data.pileup_generator import PileupGenerator
import numpy as np
import os

os.makedirs("sample_data", exist_ok=True)

gen = PileupGenerator(height=64, width=32, channels=5, seed=42)

# Generate one image per class
class_names = ["No Variant", "SNP", "Insertion", "Deletion"]
for label in range(4):
    img, _ = gen.generate(label)
    np.save(f"sample_data/class_{label}_{class_names[label]}.npy", img)
    print(f"Saved: sample_data/class_{label}_{class_names[label]}.npy (label={label})")

print("\nDone! Upload these .npy files in the Predict tab.")