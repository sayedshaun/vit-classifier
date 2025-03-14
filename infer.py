import os

from PIL import Image
from src.model import VITImageClassifier
import torch
from src.trainer import Trainer
from src.config import ModelConfig
import argparse
import json
from torchvision import transforms
from src.mapper import Mapper

def main(args: argparse.Namespace) -> None:
    with open(os.path.join(args.save_directory, "config.json"), "r") as f:
        config = json.load(f)
    model = VITImageClassifier(
        ModelConfig(
            image_size=config["image_size"],
            patch_size=config["patch_size"],
            color_channels=config["color_channels"],
            hidden_size=config["hidden_size"],
            num_heads=config["num_heads"],
            num_layers=config["num_layers"],
            norm_epsilon=config["norm_epsilon"],
            dropout=config["dropout"],
            num_class=config["num_class"],
        )
    )
    model.load_state_dict(
        torch.load(
            os.path.join(args.save_directory, "pytorch_model.pt"), 
            weights_only=True, 
            map_location=args.device), 
        strict=True
    )
    model.to(args.device)

    image = Image.open(args.image_path).convert("RGB")
    image = transforms.Resize((config["image_size"], config["image_size"]))(image)
    image = transforms.ToTensor()(image)
    image = image.unsqueeze(0)
    image = image.to(args.device)

    model.eval()
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output.logits, 1)

    result = config["label_to_class"][str(predicted.item())]
    print(f"Prediction: {result}")


parser = argparse.ArgumentParser()
parser.add_argument("--image_path", type=str, required=True)
parser.add_argument("--save_directory", type=str, required=True)
parser.add_argument("--device", type=str, default="cpu")
args = parser.parse_args()

if __name__ == "__main__":
    main(args)

    # python infer.py --image_path "logo.png" --save_directory "my_checkpoint"