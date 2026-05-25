import argparse
from config import CONFIG, DEVICE, LABEL_MAP
from data_preparation import download_data, get_dataloaders
from model import build_model, load_trained_model
from train import train_model
from evaluate import final_evaluate
from inference import predict_image


def main():
    parser = argparse.ArgumentParser(description="CIFAKE Detection Pipeline")
    parser.add_argument("mode", choices=["train", "evaluate", "infer"], help="Mode to run")
    parser.add_argument("--image", type=str, help="Path to image for inference mode")
    args = parser.parse_args()

    print(f"Using Device: {DEVICE}")

    if args.mode == "train":
        # 1. Download data (skips if exists) & Train
        download_data(CONFIG)
        model = build_model(CONFIG, DEVICE)
        train_loader, test_loader, class_to_idx = get_dataloaders(CONFIG)
        train_model(model, train_loader, test_loader, CONFIG, DEVICE, class_to_idx, LABEL_MAP)
        final_evaluate(model, test_loader, DEVICE)

    elif args.mode == "evaluate":
        # 1. Download data (skips if exists)
        download_data(CONFIG)
        _, test_loader, _ = get_dataloaders(CONFIG)

        # 2. Load model from folder
        model = load_trained_model(CONFIG, DEVICE)

        # 3. Perform test/evaluation analysis
        final_evaluate(model, test_loader, DEVICE)

    elif args.mode == "infer":
        if not args.image:
            print("Please provide an image path using --image")
            return

        # 2. Load model from folder
        model = load_trained_model(CONFIG, DEVICE)

        # 3. Perform inference
        predict_image(args.image, model, CONFIG, DEVICE, LABEL_MAP)


if __name__ == "__main__":
    main()