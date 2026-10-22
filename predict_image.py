"""Predict a variant from a pileup image and print a human-readable result.

Usage:
    python predict_image.py --model saved_models/cnn_best.pth --image sample.npy
"""
from __future__ import annotations

import argparse
import os
import warnings
warnings.filterwarnings("ignore")
import numpy as np

from backend.predict import ModelPredictor
from reporting import describe_prediction


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--image", type=str, required=True)
    args = parser.parse_args()

    predictor = ModelPredictor(config_path=args.config, model_path=args.model)

    if args.image.endswith(".npy"):
        image = np.load(args.image)
    else:
        from PIL import Image
        img = Image.open(args.image).convert("RGB")
        image = np.array(img).transpose(2, 0, 1) / 255.0

    if image.ndim == 3 and image.shape[-1] in (3, 5, 8):
        image = image.transpose(2, 0, 1)

    result = predictor.predict(image)
    print(describe_prediction(
        result["class"], result["confidence"], result["class_name"]
    ))


if __name__ == "__main__":
    main()