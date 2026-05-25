import os
from transformers import ViTForImageClassification
from peft import get_peft_model, LoraConfig, PeftModel


def build_model(config, device):
    """Loads the ViT model, applies LoRA, and moves it to the target device (used for training)."""
    print("Loading base ViT model...")
    model = ViTForImageClassification.from_pretrained(
        config["model_name"],
        num_labels=config["num_classes"],
        ignore_mismatched_sizes=True
    )

    lora_config = LoraConfig(
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    model = get_peft_model(model, lora_config)
    model = model.to(device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    print(f"LoRA Applied | Trainable %: {100 * trainable_params / total_params:.2f}%")
    return model


def load_trained_model(config, device):
    """Loads the base ViT model and applies the saved pre-trained LoRA weights."""
    print(f"Loading base model {config['model_name']}...")
    base_model = ViTForImageClassification.from_pretrained(
        config["model_name"],
        num_labels=config["num_classes"],
        ignore_mismatched_sizes=True
    )

    model_path = config["save_dir"]
    # Check if the notebook saved it inside a 'best_model' subfolder
    if os.path.exists(os.path.join(model_path, "best_model")):
        model_path = os.path.join(model_path, "best_model")

    print(f"Loading trained weights from {model_path}...")
    model = PeftModel.from_pretrained(base_model, model_path)
    model = model.to(device)

    return model