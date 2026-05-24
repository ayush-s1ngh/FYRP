import torch
from torch.amp import autocast
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, \
    confusion_matrix
from tqdm import tqdm


def final_evaluate(model, test_loader, device):
    """Runs a full evaluation on the test set and prints metrics."""
    model.eval()
    all_preds, all_labels = [], []

    eval_bar = tqdm(test_loader, desc="Final Evaluation")

    with torch.no_grad():
        for images, labels in eval_bar:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            with autocast(device_type=device.type):
                outputs = model(pixel_values=images).logits

            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average="weighted")
    recall = recall_score(all_labels, all_preds, average="weighted")
    f1 = f1_score(all_labels, all_preds, average="weighted")

    print("\n" + "=" * 60)
    print("FINAL EVALUATION REPORT")
    print("=" * 60)
    print(f" Accuracy  : {accuracy:.4f}  ({accuracy * 100:.2f}%)")
    print(f" Precision : {precision:.4f}")
    print(f" Recall    : {recall:.4f}")
    print(f" F1 Score  : {f1:.4f}")
    print("=" * 60)

    print("\n Per-Class Classification Report:\n")
    print(classification_report(all_labels, all_preds, target_names=["FAKE", "REAL"]))