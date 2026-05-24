from transformers import ViTForImageClassification
from peft import get_peft_model, LoraConfig


def build_model(config, device):
    """Loads the ViT model, applies LoRA, and moves it to the target device."""
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
        target_modules=["query", "value"],
    )

    model = get_peft_model(model, lora_config)
    model = model.to(device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    print(f"LoRA Applied | Trainable %: {100 * trainable_params / total_params:.2f}%")
    return model