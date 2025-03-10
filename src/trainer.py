import json
import statistics
from typing import Tuple, Union
import warnings
import torch
import math
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LinearLR
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import wandb
#from src.metrics import accuracy_score, f1_score, precision_score, recall_score
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
        wandb_runname: Union[str, None] = None
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
        self.wandb_runname = wandb_runname

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

        self.train_data = self._create_dataloader(self.train_data)
        self.val_data = self._create_dataloader(self.val_data)
        self.test_data = self._create_dataloader(self.test_data)


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

    @property
    def is_cuda(self) -> bool:
        # Supports both string and torch.device
        if isinstance(self.device, str):
            return self.device.lower() == "cuda"
        elif isinstance(self.device, torch.device):
            return self.device.type == "cuda"
        return False

    def train(self):
        global_step = 0

        for epoch in range(self.epochs):
            self.model.train()
            self.optimizer.zero_grad()

            for step, batch in enumerate(tqdm(self.train_data, total=len(self.train_data), desc="Training")):
                with autocast(device_type=self.device, dtype=self.precision, enabled=self.is_cuda):
                    outputs = self.step(batch)
                    loss = outputs.loss
                    train_loss = loss / self.gradient_accumulation_steps

                self.scaler.scale(train_loss).backward()

                # Gradient accumulation step
                if (step + 1) % self.gradient_accumulation_steps == 0:
                    if self.gradient_clipping is not None:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clipping)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()

                global_step += 1

                # Logging
                if self.log_and_eval_step and global_step % self.log_and_eval_step == 0:
                    logs = {"train_loss": train_loss.item() * self.gradient_accumulation_steps}
                    if self.val_data is not None:
                        val_loss, val_f1 = self.evaluate(self.val_data)
                        logs.update({"val_loss": val_loss, "val_f1": val_f1})
                        self.model.train()

                    if self.report_to_wandb:
                        wandb.log(logs, step=global_step)
                    
                    print(logs)

    def evaluate(self, dataloader: DataLoader) -> Tuple[float, float]:
        self.model.eval()
        all_y_true = []
        all_y_pred = []
        total_loss = 0.0

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                outputs = self.step(batch)
                total_loss += outputs.loss.item()
                all_y_true.append(batch["labels"].detach().cpu())
                all_y_pred.append(outputs.logits.argmax(dim=-1).detach().cpu())

        # Concatenate all batches
        y_true = torch.cat(all_y_true)
        y_pred = torch.cat(all_y_pred)
        avg_loss = total_loss / len(dataloader)
        # Convert tensors to numpy arrays for f1_score
        f1 = f1_score(y_true.numpy(), y_pred.numpy(), average="macro")
        return avg_loss, f1

    def step(self, batch):
        inputs = batch["inputs"].to(self.device)
        labels = batch["labels"].to(self.device)
        outputs = self.model(inputs, labels)
        return outputs


    def predict(self, dataset: Dataset) :
        self.model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for batch in dataset:
                inputs, label = batch
                inputs = inputs.to(self.device)
                label = label.to(self.device)
                outputs = self.model(inputs, label)
                y_pred = torch.argmax(outputs.logits, dim=1)
                y_pred.extend(y_pred.cpu().numpy())

        report = classification_report(y_true, y_pred, output_dict=True)
        if self.report_to_wandb:
            wandb.log(
                {   "test_accuracy": report["accuracy"], 
                    "test_precision": report["macro avg"]["precision"], 
                    "test_recall": report["macro avg"]["recall"], 
                    "test_f1": report["macro avg"]["f1-score"]
                }
            )
        with open("classification_report.json", "w") as f:
            json.dump(report, f, indent=4)
                

