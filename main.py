import argparse
from config import CONFIG, DEVICE, LABEL_MAP
from data_preparation import download_data, get_dataloaders
from model import build_model
from train import train_model
from evaluate import final_evaluate
from inference import predict_image

def main():
    parser = argparse.ArgumentParser(description="CIFAKE Detection Pipeline")
    parser.add_argument("mode", choices=["train", "evaluate", "infer"], help="Mode to run")
    parser.add_argument("--image", type=str, help="Path to image for inference mode")
    args = parser.parse_args()

    print(f"🖥️  Using Device: {DEVICE}")

    # Build model (we build it from scratch here; if evaluating/inferring, you'd load weights)
    # Note: If inferring/evaluating, you should technically load the saved PeftModel from CONFIG["save_dir"]
    model = build_model(CONFIG, DEVICE)

    if args.mode == "train":
        download_data(CONFIG)
        train_loader, test_loader, class_to_idx = get_dataloaders(CONFIG)
        train_model(model, train_loader, test_loader, CONFIG, DEVICE, class_to_idx, LABEL_MAP)
        final_evaluate(model, test_loader, DEVICE)

    elif args.mode == "evaluate":
        download_data(CONFIG)
        _, test_loader, _ = get_dataloaders(CONFIG)
        # Assuming model weights are loaded here from CONFIG["save_dir"]
        final_evaluate(model, test_loader, DEVICE)

    elif args.mode == "infer":
        if not args.image:
            print("Please provide an image path using --image")
            return
        # Assuming model weights are loaded here from CONFIG["save_dir"]
        predict_image(args.image, model, CONFIG, DEVICE, LABEL_MAP)

if __name__ == "__main__":
    main()