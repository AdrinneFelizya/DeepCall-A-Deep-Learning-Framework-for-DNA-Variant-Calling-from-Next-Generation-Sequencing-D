"""Basic import and construction tests for DeepCall-Real."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from config import NUM_CLASSES, IMAGE_CHANNELS
from models.cnn import VariantCNN
from models.resnet18 import VariantResNet18
from models.vgg16 import VariantVGG16
from models.transformer_caller import VariantCallerTransformer


def test_model_construction():
    for Model, kwargs in [
        (VariantCNN, {"in_channels": IMAGE_CHANNELS, "num_classes": NUM_CLASSES}),
        (VariantResNet18, {"in_channels": IMAGE_CHANNELS, "num_classes": NUM_CLASSES, "pretrained": False}),
        (VariantVGG16, {"in_channels": IMAGE_CHANNELS, "num_classes": NUM_CLASSES, "pretrained": False}),
        (VariantCallerTransformer, {"in_channels": IMAGE_CHANNELS, "num_classes": NUM_CLASSES}),
    ]:
        m = Model(**kwargs)
        x = torch.zeros(2, IMAGE_CHANNELS, 64, 32)
        out = m(x)
        assert out.shape == (2, NUM_CLASSES), (Model.__name__, out.shape)


if __name__ == "__main__":
    test_model_construction()
    print("All import/construction tests passed.")