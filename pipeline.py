import os
from src.data import ImageClassificationDataset
from src.model import VITImageClassifier
from src.trainer import Trainer
from src.config import ModelConfig
import argparse
import torch
import torchvision
from torchvision import transforms
from typing import List
from torch.utils.data import random_split

def main(args: argparse.Namespace) -> None:
    full_dataset = ImageClassificationDataset(
        root_dir=args.data_dir, 
        resize_image=args.image_size, 
        normalize=args.normalize,
        add_custom_transform=args.additional_transform
        )
    dataset = {"train": full_dataset, "val": None, "test": None}
    if args.split_data:
        from torch.utils.data import random_split
        # Compute lengths for 80/20 split
        total_length = len(full_dataset)
        train_length = int(0.8 * total_length)
        val_length = total_length - train_length
        train_dataset, val_dataset = random_split(full_dataset, [train_length, val_length])
        dataset = {"train": train_dataset, "val": val_dataset, "test": None}

    model = VITImageClassifier(
        ModelConfig(
            image_size=args.image_size,
            patch_size=args.patch_size,
            color_channels=args.color_channels,
            hidden_size=args.hidden_size,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            norm_epsilon=args.norm_epsilon,
            dropout=args.dropout,
            num_class=args.num_class
        )
    )
    trainer = Trainer(
        model=model,
        train_data=dataset["train"],
        val_data=dataset["val"],
        test_data=dataset["test"],
        batch_size=args.batch_size,
        device=args.device,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_clipping=args.gradient_clipping,
        precision=args.precision,
        log_and_eval_step=args.log_and_eval_step,
        save_steps=args.save_steps,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        shuffle_data=args.shuffle_data,
        seed=args.seed,
        report_to_wandb=args.report_to_wandb,
        wandb_project=args.wandb_project,
        wandb_runname=args.wandb_runname
    )
    trainer.train()

    if args.do_predict:
        trainer.predict(val_dataset)

def custom_transforms() -> List[transforms.Compose]:
    return [
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        transforms.RandomInvert(p=0.5),
        transforms.RandomAutocontrast(p=0.5),
        transforms.RandomEqualize(p=0.5),
        transforms.RandomGrayscale(p=0.5),
    ]

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="data", help="Path to the dataset directory.")
parser.add_argument("--image_size", type=int, default=224, help="Target image size.")
parser.add_argument("--patch_size", type=int, default=16, help="Patch size.")
parser.add_argument("--color_channels", type=int, default=3, help="Number of color channels.")
parser.add_argument("--hidden_size", type=int, default=768, help="Hidden size.")
parser.add_argument("--num_heads", type=int, default=12, help="Number of heads.")
parser.add_argument("--num_layers", type=int, default=12, help="Number of layers.")
parser.add_argument("--norm_epsilon", type=float, default=1e-5, help="Norm epsilon.")
parser.add_argument("--dropout", type=float, default=0.1, help="Dropout.")
parser.add_argument("--num_class", type=int, required=True, help="Number of classes.")
parser.add_argument("--normalize", action="store_true", help="Apply ImageNet normalization.")
parser.add_argument("--additional_transform", action="store_true", help="Apply additional transformations.")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
parser.add_argument("--epochs", type=int, default=1, help="Number of epochs.")
parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate.")
parser.add_argument("--weight_decay", type=float, default=0, help="Weight decay.")
parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps.")
parser.add_argument("--gradient_clipping", type=float, default=1.0, help="Gradient clipping.")
parser.add_argument("--precision", type=str, default="fp16", help="Precision.")
parser.add_argument("--device", type=str, default="cpu", help="Device.")
parser.add_argument("--log_and_eval_step", type=int, default=10, help="Logging steps.")
parser.add_argument("--save_steps", type=int, default=10, help="Save steps.")
parser.add_argument("--num_workers", type=int, default=1, help="Number of workers.")
parser.add_argument("--pin_memory", action="store_true", help="Pin memory.")
parser.add_argument("--shuffle_data", action="store_true", help="Shuffle data.")
parser.add_argument("--seed", type=int, default=None, help="Seed.")
parser.add_argument("--report_to_wandb", action="store_true", help="Report to wandb.")
parser.add_argument("--wandb_project", type=str, default=None, help="Wandb project name.")
parser.add_argument("--wandb_runname", type=str, default=None, help="Wandb run name.")
parser.add_argument("--split_data", action="store_true", help="Split data from training set for validation.")
parser.add_argument("--do_predict", action="store_true", help="Do prediction after training.")
args = parser.parse_args()

if __name__ == "__main__":
    main(args)