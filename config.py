"""
Configuration for DNA Variant Calling Project (synthetic mode).
Used by smoke_test.py, train.py --mode synthetic, evaluate.py --mode synthetic.
"""

import torch

# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data parameters
NUM_CLASSES = 4
CLASS_NAMES = ['No Variant', 'SNP', 'Insertion', 'Deletion']
IMAGE_CHANNELS = 5
IMAGE_HEIGHT = 64
IMAGE_WIDTH = 32
BATCH_SIZE = 64

# Training parameters
EPOCHS = 15
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

# Class weights (inverse frequency for imbalanced data)
CLASS_WEIGHTS = [1.0, 5.0, 10.0, 10.0]

# Paths
DATA_DIR = './data'
MODEL_DIR = './saved_models'
OUTPUT_DIR = './output'

# Synthetic data parameters
NUM_TRAIN_SAMPLES = 4000
NUM_VAL_SAMPLES = 800
NUM_TEST_SAMPLES = 800