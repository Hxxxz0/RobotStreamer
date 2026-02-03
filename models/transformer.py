import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LLaMAHFConfig:
    block_size: int = 78
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 768
    text_encoder_dim: int = 384

    @classmethod
    def from_name(cls, name: str):
        return cls(**llama_configs[name])


llama_configs = {
    "Normal_size": dict(n_layer=8, n_head=8, n_embd=768),
}


class LLaMAHF(nn.Module):
    def __init__(self, config: LLaMAHFConfig, input_token_dim=16) -> None:
        super().__init__()
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Linear(input_token_dim, config.n_embd),
                cond_embed=nn.Linear(config.text_encoder_dim, config.n_embd),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=RMSNorm(config.n_embd),
            )
        )
        self.out_proj = nn.Linear(config.n_embd, config.n_embd)

    def forward(self, idx: torch.Tensor, feature: torch.Tensor) -> torch.Tensor:
        if len(idx) == 0:
            token_embeddings = self.transformer.cond_embed(feature).unsqueeze(0)
        else:
            b, t, _ = idx.size()
            idx = idx.float()
            assert t <= self.config.block_size, (
                f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
            )

            token_embeddings = self.transformer.wte(idx)
            text_embeddings = self.transformer.cond_embed(feature).unsqueeze(1)
            token_embeddings = torch.cat([text_embeddings, token_embeddings], dim=1)

        x = token_embeddings
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        return self.out_proj(x)


class Block(nn.Module):
    def __init__(self, config: LLaMAHFConfig) -> None:
        super().__init__()
        self.rms_1 = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.rms_2 = RMSNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.rms_1(x))
        x = x + self.mlp(self.rms_2(x))
        return x


class CausalSelfAttention(nn.Module):
    def __init__(self, config: LLaMAHFConfig) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.block_size = config.block_size
        self.rope_cache = None

        def scaling_factor(sequence_threshold):
            return np.log2((sequence_threshold**2) - sequence_threshold)

        scale_init = scaling_factor(self.block_size)
        self.scale = nn.Parameter(torch.tensor(scale_init))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        head_size = C // self.n_head
        k = k.view(B, T, self.n_head, head_size).transpose(1, 2)
        q = q.view(B, T, self.n_head, head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_size).transpose(1, 2)

        q = F.normalize(q, p=2, dim=-1)
        k = F.normalize(k, p=2, dim=-1)

        # Dynamic RoPE cache handling to ensure correct device
        if self.rope_cache is None or self.rope_cache.device != x.device:
            self.rope_cache = build_rope_cache(
                seq_len=self.block_size,
                n_elem=self.n_embd // self.n_head,
                dtype=x.dtype,
                device=x.device,
            )

        q = apply_rope(q, self.rope_cache)
        k = apply_rope(k, self.rope_cache)

        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True, scale=self.scale.item()
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config: LLaMAHFConfig) -> None:
        super().__init__()
        hidden_dim = 4 * config.n_embd
        n_hidden = int(2 * hidden_dim / 3)
        n_hidden = ((n_hidden - 1) // 256) * 256 + 256

        self.c_fc1 = nn.Linear(config.n_embd, n_hidden, bias=False)
        self.c_fc2 = nn.Linear(config.n_embd, n_hidden, bias=False)
        self.c_proj = nn.Linear(n_hidden, config.n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.c_fc1(x)) * self.c_fc2(x)
        x = self.c_proj(x)
        return x


class RMSNorm(nn.Module):
    def __init__(self, size: int, dim: int = -1, eps: float = 1e-5) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(size))
        self.eps = eps
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm_x = torch.mean(x * x, dim=self.dim, keepdim=True)
        x_normed = x * torch.rsqrt(norm_x + self.eps)
        return self.scale * x_normed


def build_rope_cache(seq_len: int, n_elem: int, dtype: torch.dtype, device: torch.device, base: int = 10000) -> torch.Tensor:
    theta = 1.0 / (base ** (torch.arange(0, n_elem, 2, dtype=dtype, device=device) / n_elem))
    seq_idx = torch.arange(seq_len, dtype=dtype, device=device)
    idx_theta = torch.outer(seq_idx, theta)
    dtypes_requiring_casting = [torch.float16, torch.bfloat16, torch.int8]
    working_dtype = torch.float32 if dtype in dtypes_requiring_casting else dtype
    complex_dtype = torch.complex32 if dtype in dtypes_requiring_casting else torch.complex64
    cache = torch.polar(
        torch.ones_like(idx_theta).to(working_dtype), idx_theta.to(working_dtype)
    ).to(complex_dtype)
    return cache


def apply_rope(x: torch.Tensor, rope_cache: torch.Tensor) -> torch.Tensor:
    x = x.transpose(1, 2)
    T = x.size(1)
    rope_cache = rope_cache[:T]
    xc = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    rope_cache = rope_cache.view(1, xc.size(1), 1, xc.size(3))
    x_out = torch.view_as_real(xc * rope_cache).flatten(3)
    return x_out.transpose(1, 2).type_as(x)
