"""
This file contains supplimentary Generic Modules to be used throughout the Repo.

    - MLP: layernorm / linear / gelu / linear, keeps d_model
    - Attention: nn.MultiheadAttention with optional layernorm on q, k, v
    - Residual: module(inpt) + residual, inpt is unpacked when it is a tuple
    - CrossAttentionLayer / SelfAttentionLayer: residual attention then residual MLP (used by LosAlamosSensiever.py)
    - FourierNeuralOperator: placeholder, not implemented
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()

        self.mlp = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
        )

    def forward(self, x):
        return self.mlp(x)

class Attention(nn.Module):

    def __init__(self,
                 num_q_channels: int,
                 num_kv_channels: int,
                 num_heads: int,
                 dropout: float,
                 norm: bool):

        super().__init__()
        self.norm = norm

        if self.norm:
            self.q_norm = nn.LayerNorm(num_q_channels)
            self.k_norm = nn.LayerNorm(num_kv_channels)
            self.v_norm = nn.LayerNorm(num_kv_channels)

        self.attention = nn.MultiheadAttention(
                            embed_dim=num_q_channels,
                            num_heads=num_heads,
                            kdim=num_kv_channels,
                            vdim=num_kv_channels,
                            dropout=dropout,
                            batch_first=True,
                        )

    def forward(self, x_q, x_k, x_v, pad_mask=None, attn_mask=None):

        if self.norm:
            x_q = self.q_norm(x_q)
            x_k = self.k_norm(x_k)
            x_v = self.v_norm(x_v)

        return self.attention(x_q, x_k, x_v, key_padding_mask=pad_mask, attn_mask=attn_mask)[0]

def Residual(module: nn.Module, residual: torch.Tensor, inpt: tuple | torch.Tensor):
    if isinstance(inpt, tuple):
        out =  module.forward(*inpt)

    elif isinstance(inpt, torch.Tensor):
        out =  module.forward(inpt)
    else:
        raise TypeError(f"Could not assertain type of inpt: {type(inpt)}")

    assert(out.shape == residual.shape)
    return out + residual

class CrossAttentionLayer(nn.Module):
    def __init__(
        self,
        num_q_channels:         int,
        num_kv_channels:        int,
        num_heads:              int,
        dropout:                float,
    ):
        super().__init__()

        self.attention = Attention(
                            num_q_channels=num_q_channels,
                            num_kv_channels=num_kv_channels,
                            num_heads=num_heads,
                            dropout=dropout,
                            norm=True,
                        )
        
        self.mlp = MLP(num_q_channels)

    def forward(self, x_q, x_kv, pad_mask=None):
        x_q = Residual(self.attention, x_q, (x_q, x_kv, x_kv, pad_mask))
        x_q = Residual(self.mlp, x_q, x_q)
        return x_q

class SelfAttentionLayer(nn.Module):
    def __init__(
        self,
        num_q_channels:         int,
        num_kv_channels:        int,
        num_heads:              int,
        dropout:                float,
    ):
        super().__init__()

        self.attention = Attention(
                            num_q_channels=num_q_channels,
                            num_kv_channels=num_kv_channels,
                            num_heads=num_heads,
                            dropout=dropout,
                            norm=True,
                        )
        
        self.mlp = MLP(num_q_channels)

    def forward(self, x):
        x = Residual(self.attention, x, (x, x, x))
        x = Residual(self.mlp, x, x)
        return x

class FourierNeuralOperator(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        pass