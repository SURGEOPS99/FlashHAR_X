"""FlashHAR-X research model. The only attention backend switch is the kernel."""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F


def sinusoidal_position_encoding(length, width):
    pos = torch.arange(length).float().unsqueeze(1)
    div = torch.exp(torch.arange(0, width, 2).float() * (-math.log(10000.0) / width))
    pe = torch.zeros(length, width)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe.unsqueeze(0)


class MultiScaleStem(nn.Module):
    def __init__(self, width=128, multiscale=True):
        super().__init__()
        kernels = [3, 7, 15] if multiscale else [7]
        branch_width = 48 if multiscale else 128
        self.branches = nn.ModuleList([
            nn.Sequential(nn.Conv1d(9, branch_width, k, padding=k//2),
                          nn.GroupNorm(8, branch_width), nn.GELU()) for k in kernels
        ])
        self.merge = nn.Sequential(nn.Conv1d(len(kernels)*branch_width, width, 1),
                                   nn.GroupNorm(8, width), nn.GELU())

    def forward(self, x):
        return self.merge(torch.cat([branch(x) for branch in self.branches], dim=1)).transpose(1, 2)


class Attention(nn.Module):
    def __init__(self, width=128, heads=4, backend='triton_flash', dropout=0.1):
        super().__init__()
        assert width % heads == 0
        self.width, self.heads, self.backend = width, heads, backend
        self.qkv = nn.Linear(width, 3*width)
        self.proj = nn.Linear(width, width)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.heads, c//self.heads)
        q, k, v = [u.contiguous() for u in qkv.permute(2, 0, 3, 1, 4).unbind(0)]
        if self.backend == 'triton_flash':
            from flash_attention_triton import flash_attention
            # CUDA autocast should make Q/K/V fp16; fail loudly otherwise.
            y = flash_attention(q, k, v)
        elif self.backend == 'torch_sdpa':
            y = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        else:
            raise ValueError(f'Unknown backend {self.backend}')
        y = y.transpose(1, 2).reshape(b, n, c)
        return self.drop(self.proj(y))


class TransformerBlock(nn.Module):
    def __init__(self, width=128, backend='triton_flash', dropout=0.12):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.attn = Attention(width, heads=4, backend=backend, dropout=dropout)
        self.mlp = nn.Sequential(nn.Linear(width, 2*width), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(2*width, width),
                                 nn.Dropout(dropout))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class FlashHARX(nn.Module):
    def __init__(self, backend='triton_flash', width=128, depth=2,
                 use_features=True, multiscale=True, dropout=0.12):
        super().__init__()
        self.backend, self.use_features = backend, use_features
        self.stem = MultiScaleStem(width, multiscale)
        self.register_buffer('pos', sinusoidal_position_encoding(128, width), persistent=False)
        self.blocks = nn.Sequential(*[
            TransformerBlock(width, backend, dropout) for _ in range(depth)
        ]) if backend != 'cnn_only' else nn.Identity()
        self.norm = nn.LayerNorm(width)
        if use_features:
            self.feature_mlp = nn.Sequential(nn.LayerNorm(561), nn.Linear(561, 128),
                                             nn.GELU(), nn.Dropout(0.2))
        self.head = nn.Sequential(nn.Linear(2*width + (128 if use_features else 0), 192),
                                  nn.GELU(), nn.Dropout(dropout), nn.Linear(192, 6))

    def forward(self, raw, features):
        x = self.stem(raw)
        x = x + self.pos.to(dtype=x.dtype)
        x = self.norm(self.blocks(x))
        pooled = torch.cat([x.mean(1), x.amax(1)], dim=-1)
        if self.use_features:
            pooled = torch.cat([pooled, self.feature_mlp(features)], dim=-1)
        return self.head(pooled)


def augment_raw(x, noise_std=0.018, scaling=0.07, mask_prob=0.2):
    """Training-time sensor augmentations; never change labels or create fake records."""
    if noise_std:
        x = x + torch.randn_like(x) * noise_std
    if scaling:
        x = x * (1 + torch.randn(x.shape[0], x.shape[1], 1, device=x.device) * scaling)
    if mask_prob:
        mask = (torch.rand(x.shape[0], x.shape[1], 1, device=x.device) < mask_prob*0.13)
        x = x.masked_fill(mask, 0)
    return x
