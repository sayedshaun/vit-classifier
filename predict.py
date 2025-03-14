import os
from src.model import VITImageClassifier
from src.data import ImageClassificationDataset
import torch
from src.trainer import Trainer
from src.config import ModelConfig
import argparse
import json


def main(args: argparse.Namespace) -> None:
    with open(os.path.join(args.saved_dir, "config.json"), "r") as f:
        config = json.load(f)

    dataset = ImageClassificationDataset(
        root_dir=args.data_dir,
        resize_image=config["image_size"],
        normalize=True,
        add_custom_transform=None,
    )

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
            os.path.join(args.saved_dir, "pytorch_model.pt"), 
            weights_only=True, 
            map_location=args.device), 
        strict=True
    )

    trainer = Trainer(model=model)
    trainer.predict(dataset)  


parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, required=True)
parser.add_argument("--saved_dir", type=str, required=True)
parser.add_argument("--device", type=str, default="cpu")
args = parser.parse_args()

if __name__ == "__main__":
    main(args)