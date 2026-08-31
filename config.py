DATASET_PATH = "dataset"

# Image
IMAGE_SIZE = 224

# Classes
NUM_CLASSES = 2

CLASS_NAMES = [
    "Covid",
    "Healthy"
]

# Training
BATCH_SIZE = 32
EPOCHS = 50

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-4

DEVICE = "cuda"

NUM_WORKERS = 0

# Early Stopping
PATIENCE = 5

# Random Seed
SEED = 42

# Mixed Precision
USE_AMP = True

# Save Directory
CHECKPOINT_DIR = "checkpoints"

LOG_DIR = "runs"

# Fusion training
CCAFN_LEARNING_RATE = 5e-5






