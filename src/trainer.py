import json
import os
import statistics
from typing import Tuple, Union
import warnings
import numpy as np
import torch
import math
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LinearLR
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import wandb
from sklearn.metrics import classification_report, f1_score

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        device: Union[str, torch.device] = "cpu",
        learning_rate: float = 1e-3,
        weight_decay: float = 0,
        epochs: int = 1,
        batch_size: int = 1,
        gradient_accumulation_steps: int = 1,
        gradient_clipping: float = 1.0,
        precision: str = "fp16",
        log_and_eval_step: int = 10,
        save_steps: int = 10,
        num_workers: int = 1,
        pin_memory: bool = True,
        shuffle_data: bool = True,
        seed: Union[int, None] = None,
        train_data: Union[Dataset, None] = None,
        val_data: Union[Dataset, None] = None,
        test_data: Union[Dataset, None] = None,
        optimizer: Union[Optimizer, None] = None,
        report_to_wandb: Union[bool, None] = None,
        wandb_project: Union[str, None] = None,
        save_directory: Union[str, None] = None
        ) -> None:
        self.model = model
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.epochs = epochs
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.gradient_clipping = gradient_clipping
        self.precision = torch.float16 if precision == "fp16" else torch.float32
        self.seed = seed
        self.log_and_eval_step = log_and_eval_step
        self.save_steps = save_steps
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.shuffle_data = shuffle_data

        self.report_to_wandb = report_to_wandb
        self.wandb_project = wandb_project
        self.save_directory = save_directory
        self.model.to(self.device)
        self.scaler = GradScaler()

        if optimizer is None:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(), 
                lr=self.learning_rate, 
                weight_decay=self.weight_decay
            )
        else:
            self.optimizer = optimizer

        if self.seed is not None:
            torch.manual_seed(self.seed)
            if hasattr(torch.cuda, "manual_seed"):
                torch.cuda.manual_seed(self.seed)
            torch.backends.cudnn.deterministic = True
        self.train_dataset = self.train_data
        self.train_data = self._create_dataloader(self.train_data)
        self.val_data = self._create_dataloader(self.val_data)

        if self.report_to_wandb:
            wandb.init(project=self.wandb_project, name=self.save_directory)

    def _create_dataloader(self, dataset: Union[Dataset, DataLoader, None]) -> Union[DataLoader, None]:
        if dataset is None:
            return None
        else:
            if isinstance(dataset, DataLoader):
                return dataset
            else:
                return DataLoader(
                    dataset=dataset, 
                    batch_size=self.batch_size,
                    shuffle=self.shuffle_data,
                    num_workers=self.num_workers, 
                    pin_memory=self.pin_memory
                )

    def _save_training_config(self, save_path: str) -> None:
        config_dict = {
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'gradient_accumulation_steps': self.gradient_accumulation_steps,
            'gradient_clipping': self.gradient_clipping,
            'precision': str(self.precision),
            'log_and_eval_step': self.log_and_eval_step,
            'save_steps': self.save_steps,
            'num_workers': self.num_workers,
            'pin_memory': self.pin_memory,
            'shuffle_data': self.shuffle_data,
            'seed': self.seed,
            'report_to_wandb': self.report_to_wandb,
            'wandb_project': self.wandb_project,
            'save_directory': self.save_directory
        }
        config_dict.update(self.model.config.__dict__)
        config_dict["label_to_class"] = self.train_dataset.dataset.label_to_class

        with open(os.path.join(save_path, "config.json"), "w") as f:
            json.dump(config_dict, f, indent=4)

        return config_dict

    @property
    def is_cuda(self) -> bool:
        # Supports both string and torch.device
        if isinstance(self.device, str):
            return self.device.lower() == "cuda"
        elif isinstance(self.device, torch.device):
            return self.device.type == "cuda"
        return False

    def train(self):
        os.makedirs(self.save_directory, exist_ok=True)
        config_dict = self._save_training_config(self.save_directory)
        if self.report_to_wandb:
            wandb.config.update(config_dict)

        global_step = 0
        best_f1 = float("-inf")
        with tqdm(total=self.epochs * len(self.train_data)) as pbar:
            for epoch in range(self.epochs):
                self.model.train()
                self.optimizer.zero_grad()

                for batch in self.train_data:
                    pbar.set_description("Training")
                    with autocast(device_type=self.device, dtype=self.precision, enabled=self.is_cuda):
                        outputs = Trainer.step(self.model, batch, self.device)
                        loss = outputs.loss
                        train_loss = loss / self.gradient_accumulation_steps

                    self.scaler.scale(train_loss).backward()

                    # Gradient accumulation step
                    if (global_step + 1) % self.gradient_accumulation_steps == 0:
                        if self.gradient_clipping is not None:
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clipping)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.optimizer.zero_grad()

                    global_step += 1

                    # Logging
                    if self.log_and_eval_step and global_step % self.log_and_eval_step == 0:
                        pbar.set_description("Evaluating")
                        logs = {"train_loss": train_loss.item() * self.gradient_accumulation_steps}
                        if self.val_data is not None:
                            val_loss, val_f1 = self.evaluate(self.model, self.val_data, self.device)
                            logs.update({"val_loss": val_loss, "val_f1": val_f1})
                            if val_f1 > best_f1:
                                best_f1 = val_f1
                                if self.save_steps is not None and global_step % self.save_steps == 0:
                                    pbar.set_description("Saving Checkpoint")
                                    Trainer.save_checkpoint(self.model, self.optimizer, self.save_directory)
                            self.model.train()
                        else:
                            logs.update({"val_loss": float("nan"), "val_f1": float("nan")})
                            if self.save_steps is not None and global_step % self.save_steps == 0:
                                pbar.set_description("Saving Checkpoint")
                                Trainer.save_checkpoint(self.model, self.optimizer, self.save_directory)

                        if self.report_to_wandb:
                            wandb.log(logs, step=global_step, sync=True)
                        
                        logs.update({"ckpt_at": f'f1:{best_f1:.3f}'})
                        pbar.set_postfix(logs)
                    pbar.update(1)

        if self.report_to_wandb:
            wandb.finish()


    @staticmethod
    def save_checkpoint(model: nn.Module, optimizer: Optimizer, save_directory: str) -> None:
        torch.save(model.state_dict(), f"{save_directory}/pytorch_model.pt")
        torch.save(optimizer.state_dict(), f"{save_directory}/optimizer.pt")

    @staticmethod
    def evaluate(model: nn.Module, dataloader: DataLoader, device) -> Tuple[float, float]:
        model.eval()
        all_y_true = []
        all_y_pred = []
        total_loss = 0.0

        with torch.no_grad():
            for batch in dataloader:
                outputs = Trainer.step(model, batch, device)
                total_loss += outputs.loss.item()
                all_y_true.append(batch["labels"].detach().cpu())
                all_y_pred.append(outputs.logits.argmax(dim=-1).detach().cpu())

        # Concatenate all batches
        y_true = torch.cat(all_y_true)
        y_pred = torch.cat(all_y_pred)
        avg_loss = total_loss / len(dataloader)
        # Convert tensors to numpy arrays for f1_score
        f1 = f1_score(y_true.tolist(), y_pred.tolist(), average="macro")
        return avg_loss, float(format(f1, ".3f"))

    @staticmethod
    def step(model: nn.Module, batch: dict, device):
        inputs = batch["inputs"].to(device)
        labels = batch["labels"].to(device)
        outputs = model(inputs, labels)
        return outputs


    def predict(self, dataset: Dataset) -> dict:
        dataloader = self._create_dataloader(dataset)
        self.model.eval()
        all_y_true, all_y_pred = [], []
        with torch.no_grad():
            for batch in tqdm(dataloader):
                inputs, label = batch["inputs"], batch["labels"]
                inputs = inputs.to(self.device)
                label = label.to(self.device)
                outputs = self.model(inputs, label)
                y_pred = torch.argmax(outputs.logits, dim=1)
                all_y_pred.extend(y_pred.cpu().numpy())
                all_y_true.extend(label.cpu().numpy())

        if self.save_directory is None:
            dict_report = classification_report(
                y_true=all_y_true, 
                y_pred=all_y_pred, 
                output_dict=True,
                zero_division=0,
                digits=4,
                target_names=dataset.label_to_class.values()
                )

            with open("test_report.txt", "w") as f:
                f.write("=" * 27 + " TEST " + "=" * 26 + "\n")
                f.write(txt_report)
        else:        
            txt_report = classification_report(
                y_true=all_y_true, 
                y_pred=all_y_pred, 
                output_dict=False,
                zero_division=0,
                digits=4,
                target_names=self.train_dataset.dataset.label_to_class.values()
                )

            if self.report_to_wandb:
                wandb.log(
                    {   "test_accuracy": dict_report["accuracy"], 
                        "test_precision": dict_report["macro avg"]["precision"], 
                        "test_recall": dict_report["macro avg"]["recall"], 
                        "test_f1": dict_report["macro avg"]["f1-score"]
                    }
                )

            with open(os.path.join(self.save_directory, "test_report.txt"), "w") as f:
                f.write("=" * 27 + " TEST " + "=" * 26 + "\n")
                f.write(txt_report)

                

