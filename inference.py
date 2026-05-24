import os
import torch
from torchvision import transforms
from PIL import Image
from torch.amp import autocast


def predict_image(image_path, model, config, device, label_map):
    """Runs a single image through the model and returns predictions."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    inference_transform = transforms.Compose([
        transforms.Resize((config["image_size"], config["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image = Image.open(image_path).convert("RGB")
    input_tensor = inference_transform(image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        with autocast(device_type=device.type):
            outputs = model(pixel_values=input_tensor).logits

    probs = torch.softmax(outputs, dim=1)
    confidence, predicted_idx = torch.max(probs, dim=1)

    predicted_label = label_map[predicted_idx.item()]
    confidence_pct = confidence.item() * 100

    print(f"\n Image       : {os.path.basename(image_path)}")
    print(f"   Prediction  : {'FAKE' if predicted_label == 'FAKE' else 'REAL'}")
    print(f"   Confidence  : {confidence_pct:.2f}%")

    return {
        "label": predicted_label,
        "confidence": round(confidence_pct, 2)
    }