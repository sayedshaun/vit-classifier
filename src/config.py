from dataclasses import dataclass
from typing import Union


@dataclass
class ModelConfig:
    image_size: Union[int, None] = None
    patch_size: Union[int, None] = None
    color_channels: Union[int, None] = None
    hidden_size: Union[int, None] = None
    num_heads: Union[int, None] = None
    num_layers: Union[int, None] = None
    norm_epsilon: Union[float, None] = None
    dropout: Union[float, None] = None
    num_class: Union[int, None] = None