import torch

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global Configuration Dictionary
CONFIG = {
    "data_dir": "./data/CIFAKE",  # Local dataset path
    "batch_size": 128,
    "num_workers": 4,
    "num_classes": 2,
    "image_size": 224,
    "lr": 1e-4,
    "epochs": 5,
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.1,
    "save_dir": "./saved_model",
    "model_name": "google/vit-base-patch16-224",
}

# Label mapping
LABEL_MAP = {0: "FAKE", 1: "REAL"}