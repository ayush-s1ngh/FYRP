import os
import json
import time
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm


def evaluate_epoch(model, loader, criterion, device, epoch, epochs):
    """Validates the model at the end of an epoch."""
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0

    val_bar = tqdm(loader, desc=f"   📊 Validating Epoch {epoch}/{epochs}", leave=False)

    with torch.no_grad():
        for images, labels in val_bar:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            with autocast(device_type=device.type):
                outputs = model(pixel_values=images).logits
                loss = criterion(outputs, labels)

            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted")

    return avg_loss, acc, f1


def train_model(model, train_loader, test_loader, config, device, class_to_idx, label_map):
    """Main training loop."""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config["lr"])
    scaler = GradScaler(device.type)

    epochs = config["epochs"]
    print(f"Starting Training for {epochs} Epochs...")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        start_time = time.time()

        batch_bar = tqdm(train_loader, desc=f"   🔁 Epoch {epoch}/{epochs} Training", leave=False)

        for images, labels in batch_bar:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type=device.type):
                outputs = model(pixel_values=images).logits
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total
        train_loss = total_loss / len(train_loader)
        epoch_time = time.time() - start_time

        val_loss, val_acc, val_f1 = evaluate_epoch(model, test_loader, criterion, device, epoch, epochs)

        print(f"Epoch [{epoch}/{epochs}] — {epoch_time:.1f}s")
        print(f"Train  → Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
        print(f"Val    → Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f}")

    # Save Model
    os.makedirs(config["save_dir"], exist_ok=True)
    model.save_pretrained(config["save_dir"])

    config_path = os.path.join(config["save_dir"], "training_config.json")
    save_config = {**config, "label_map": label_map, "class_to_idx": class_to_idx}
    with open(config_path, "w") as f:
        json.dump(save_config, f, indent=4)

    print(f"Model and config saved to {config['save_dir']}")