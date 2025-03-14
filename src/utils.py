import os
from typing import NamedTuple, Union

import torch


class Mapper:
    def __init__(self, data_dir):
        self.process(data_dir)

    def process(self, data_dir):
        self.data_dir = data_dir
        classes = sorted(os.listdir(data_dir))
        self.class_to_label = {cls: idx for idx, cls in enumerate(classes)}
        self.label_to_class = {v: k for k, v in self.class_to_label.items()}

    def index_to_class(self, index):
        return self.label_to_class[index]

    def class_to_index(self, class_name):
        return self.class_to_label[class_name]
    

def accuracy_score(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.FloatTensor:
    return (y_true == y_pred).float().mean()


def precision_score(y_true: torch.Tensor, y_pred: torch.Tensor, average: str = "macro") -> torch.FloatTensor:
    """
    Precision = True Positives / (True Positives + False Positives)
    """
    classes = torch.unique(y_true)
    epsilon = 1e-8 # to avoid division by zero
    true_positives = torch.zeros(len(classes), device=y_true.device)
    false_positives = torch.zeros(len(classes), device=y_true.device)
    for i, c in enumerate(classes):
        true_positives[i] = ((y_true == c) & (y_pred == c)).sum()
        false_positives[i] = ((y_true != c) & (y_pred == c)).sum()
    
    if average == "macro":
        return (true_positives / (true_positives + false_positives + epsilon)).mean()
    elif average == "binary":
        return true_positives / (true_positives + false_positives + epsilon)


def recall_score(y_true: torch.Tensor, y_pred: torch.Tensor, average: str = "macro") -> torch.FloatTensor:
    """
    Recall = True Positives / (True Positives + False Negatives)
    """
    classes = torch.unique(y_true)
    epsilion = 1e-8 # to avoid division by zero
    true_positives = torch.zeros(len(classes)).to(y_true.device)
    false_negatives = torch.zeros(len(classes)).to(y_true.device)
    for i, c in enumerate(classes):
        true_positives[i] = ((y_true == c) & (y_pred == c)).sum()
        false_negatives[i] = ((y_true == c) & (y_pred != c)).sum()
    
    if average == "macro":
        return (true_positives / (true_positives + false_negatives + epsilion)).mean()
    elif average == "binary":
        return true_positives / (true_positives + false_negatives + epsilion)


def f1_score(y_true: torch.Tensor, y_pred: torch.Tensor, average: str = "macro") -> torch.FloatTensor:
    """
    F1 = 2 * (Precision * Recall) / (Precision + Recall)
    """
    precision = precision_score(y_true, y_pred, average=average)
    recall = recall_score(y_true, y_pred, average=average)
    return 2 * (precision * recall) / (precision + recall)


class ModelOutput(NamedTuple):
    logits: Union[torch.Tensor, None] = None
    loss: Union[torch.Tensor, None] = None