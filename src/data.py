import torch
from torchvision import transforms
import os
from PIL import Image
import numpy as np
import random
from typing import List, Optional, Tuple, Union


class ImageClassificationDataset(torch.utils.data.Dataset):
    """
    ### Args:
        image_dir (str): Path to the dataset directory.
        resize_image (int): Target image size (default: 224).
        normalize (bool): Apply ImageNet normalization (default: False).

    ### Structure:
    ```
        root_dir/
            class_0/
                image_0.jpg
                image_1.jpg
                ...
            class_1/
                image_0.jpg
                image_1.jpg
                ...
            class_2/
                image_0.jpg
                image_1.jpg
            ...
    ```
    ### Example:
    ```python
    from src.data import ImageClassificationDataset
    from torchvision import transforms

    custom_transform = [
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        transforms.RandomInvert(p=0.5),
        transforms.RandomPosterize(bits=4, p=0.5),
        transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
        transforms.RandomSolarize(threshold=192, p=0.5)
    ]
    dataset = ImageClassificationDataset(
        root_dir="/path/to/dataset", 
        resize_image=224, 
        normalize=True,
        add_custom_transform=custom_transform
    )
    ```
    """
    def __init__(
            self, 
            root_dir: str, 
            resize_image: int, 
            normalize: bool = False, 
            add_custom_transform: Union[List[object], None] = None) -> None:
        
        self.image_dir = root_dir
        self.resize_image = resize_image
        self.normalize = normalize
        self.add_custom_transform = add_custom_transform
        
        classes = sorted(os.listdir(root_dir))
        self.class_to_label = {cls: idx for idx, cls in enumerate(classes)}
        self.label_to_class = {v: k for k, v in self.class_to_label.items()}

        self.images, self.labels = [], []
        for cls_name, cls_idx in self.class_to_label.items():
            cls_dir = os.path.join(root_dir, cls_name)
            if os.path.isdir(cls_dir):
                for img_name in os.listdir(cls_dir):
                    img_path = os.path.join(cls_dir, img_name)
                    self.images.append(img_path)
                    self.labels.append(cls_idx)

    def label_to_class(self, label: Optional[torch.Tensor]):
        """Convert label index to class name."""
        for cls_name, cls_idx in self.class_to_label.items():
            if cls_idx == label:
                return cls_name

    def transform_image(self, image: Image.Image) -> torch.FloatTensor:
        """Applies image transformations."""
        transform_list = [
            transforms.Resize((self.resize_image, self.resize_image)),
            transforms.ToTensor(),
        ]
        if self.add_custom_transform:
            transform_list += self.add_custom_transform
        if self.normalize:
            transform_list.append(
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            )

        transform = transforms.Compose(transform_list)
        return transform(image)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.FloatTensor, torch.Tensor]:
        img_path = self.images[idx]
        label = self.labels[idx]
        
        image = Image.open(img_path).convert("RGB")
        image = self.transform_image(image)
        return {"inputs": image, "labels": torch.tensor(label, dtype=torch.long)}