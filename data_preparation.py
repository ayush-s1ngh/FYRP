import os
import shutil
import kagglehub
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def download_data(config):
    """Downloads the CIFAKE dataset and moves it to the local data directory."""
    target_dir = config["data_dir"]

    if os.path.exists(target_dir) and len(os.listdir(target_dir)) > 0:
        print(f"Dataset already exists at {target_dir}")
        return

    print("Downloading dataset via kagglehub...")
    cache_path = kagglehub.dataset_download("birdy654/cifake-real-and-ai-generated-synthetic-images")

    print(f"Moving dataset from cache to {target_dir}...")
    os.makedirs(os.path.dirname(target_dir), exist_ok=True)
    shutil.copytree(cache_path, target_dir, dirs_exist_ok=True)
    print("Dataset ready!")


def get_dataloaders(config):
    """Creates and returns training and testing dataloaders."""
    train_transform = transforms.Compose([
        transforms.Resize((config["image_size"], config["image_size"])),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((config["image_size"], config["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dir = os.path.join(config["data_dir"], "train")
    test_dir = os.path.join(config["data_dir"], "test")

    train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=test_transform)

    is_multi_worker = config["num_workers"] > 0
    prefetch = 2 if is_multi_worker else None
    persistent = True if is_multi_worker else False

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=True,
        prefetch_factor=prefetch,
        persistent_workers=persistent
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=True,
        prefetch_factor=prefetch,
        persistent_workers=persistent
    )

    return train_loader, test_loader, train_dataset.class_to_idx