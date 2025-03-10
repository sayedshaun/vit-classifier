import math
import torch
from torch import nn
from typing import NamedTuple, Tuple, Union
import torch.nn.functional as F
from src.config import ModelConfig



class Attention(nn.Module):
    def __init__(self,  hidden_size: int, num_heads: int) -> None:
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        assert self.head_dim * num_heads == hidden_size, "hidden_size must be divisible by num_heads"
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size,hidden_size, bias=True)

    def forward(self, Q:torch.Tensor, K:torch.Tensor, V:torch.Tensor, mask: Union[torch.Tensor, None]=None)->torch.Tensor:
        N, L, D = Q.shape
        Q, K, V = self.q_proj(Q), self.k_proj(K), self.v_proj(V)
        Q = Q.view(N, L, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(N, L, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(N, L, self.num_heads, self.head_dim).transpose(1, 2)
        
        score = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            score = score.masked_fill(mask == 0, float("-inf"))
        weights = F.softmax(score, dim=-1)
        attention = torch.matmul(weights, V)

        output = attention.transpose(1, 2).contiguous().view(N, L, D)
        return self.out_proj(output)


class FeedForward(torch.nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, dropout: float) -> None:
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self.fc1(X)
        X = self.dropout(X)
        X = self.act(X)
        X = self.dropout(X)
        X = self.fc2(X)
        return X


class TransformerEncoderBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, norm_epsilon: float, dropout: float) -> None:
        super(TransformerEncoderBlock, self).__init__()
        self.mha = Attention(hidden_size, num_heads)
        self.norm_1 = nn.LayerNorm(hidden_size, norm_epsilon)
        self.norm_2 = nn.LayerNorm(hidden_size, norm_epsilon)
        self.mlp = FeedForward(hidden_size, hidden_size * 4, dropout)

    def forward(self, X:torch.Tensor, padding_mask: Union[torch.Tensor, None] = None)->torch.Tensor:
        attention= self.mha(X, X, X, padding_mask)
        attention = self.norm_1(attention + X)
        output = self.mlp(attention)
        return self.norm_2(output + attention)


class PatchEmbedding(nn.Module):
    def __init__(self, img_size: int, patch_size: int, in_channels: int, embed_dim: int) -> None:
        super(PatchEmbedding, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.flatten = nn.Flatten(2, 3)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        B, C, H, W = X.shape
        X = self.proj(X)  # (B, embed_dim, H/patch_size, W/patch_size)
        X = self.flatten(X)  # (B, embed_dim, num_patches)
        return X.transpose(1, 2)  # (B, num_patches, embed_dim)


class ModelOutput(NamedTuple):
    logits: Union[torch.Tensor, None] = None
    loss: Union[torch.Tensor, None] = None

class VITImageClassifier(torch.nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super(VITImageClassifier, self).__init__()
        self.config = config
        self.patch_embed = PatchEmbedding(
            img_size=config.image_size,
            patch_size=config.patch_size,
            in_channels=config.color_channels,
            embed_dim=config.hidden_size
        )
        self.pos_embed = torch.nn.Parameter(torch.randn(1, self.patch_embed.num_patches + 1, config.hidden_size))
        self.cls_token = torch.nn.Parameter(torch.randn(1, 1, config.hidden_size))
        self.blocks = torch.nn.ModuleList([
            TransformerEncoderBlock(
                hidden_size=config.hidden_size, 
                num_heads=config.num_heads,
                norm_epsilon=config.norm_epsilon,
                dropout=config.dropout) for _ in range(config.num_layers)
        ])
        self.norm = nn.LayerNorm(config.hidden_size, config.norm_epsilon)
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_class)

    def forward(self, inputs: torch.Tensor, labels: Union[torch.Tensor, None] = None) -> ModelOutput:
        B = inputs.shape[0]
        inputs = self.patch_embed(inputs)  # (B, num_patches, embed_dim)
        # Add class token
        cls_token = self.cls_token.expand(B, -1, -1)
        inputs = torch.cat((cls_token, inputs), dim=1)  # (B, num_patches+1, embed_dim)
        # Add positional embedding
        inputs = inputs + self.pos_embed
        inputs = self.dropout(inputs)
        # Process through transformer blocks
        for block in self.blocks:
            inputs = block(inputs)
        # Classification
        inputs = self.norm(inputs)
        logits = self.classifier(inputs[:, 0])
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            return ModelOutput(logits=logits, loss=loss)
        else:
            return ModelOutput(logits=logits)