# src/audio_encoders/lora.py
# Lightweight LoRA utilities without external deps.
# IMPORTANT: do NOT create cyclic Module references (causes RecursionError in named_modules).

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LoRAConfig:
    r: int = 8
    alpha: float = 16.0
    dropout: float = 0.05
    target_keywords: Tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "out_proj")


class LoRALinear(nn.Module):
    """
    Wraps an existing nn.Linear and adds a trainable low-rank update:
        W' = W + (alpha/r) * (B @ A)
    where:
        A: (r, in_features)
        B: (out_features, r)

    NOTE:
      - We keep the original Linear as a submodule (self.base).
      - We MUST NOT attach the wrapper back onto base (would create a cycle).
    """
    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError("LoRALinear expects an nn.Linear base module")

        self.base = base
        self.r = int(r)
        self.alpha = float(alpha)
        self.scaling = self.alpha / max(1, self.r)
        self.drop = nn.Dropout(p=float(dropout)) if dropout and dropout > 0 else nn.Identity()

        in_f = base.in_features
        out_f = base.out_features

        # LoRA params
        self.lora_A = nn.Parameter(torch.zeros(self.r, in_f))
        self.lora_B = nn.Parameter(torch.zeros(out_f, self.r))

        # Init: A small random, B zeros => start as no-op
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)

        # Freeze base weights by default (caller can override)
        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base linear
        y = self.base(x)
        # LoRA update
        # x: (..., in_features)
        # A: (r, in_features) => x @ A.T => (..., r)
        # B: (out_features, r) => (..., r) @ B.T => (..., out_features)
        dx = self.drop(x)
        lora = (dx @ self.lora_A.t()) @ self.lora_B.t()
        return y + self.scaling * lora


def _matches(name: str, keywords: Iterable[str]) -> bool:
    name = name.lower()
    return any(k.lower() in name for k in keywords)


@torch.no_grad()
def _count_params(m: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in m.parameters())
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    return int(trainable), int(total)


def inject_lora_into_hubert(
    hubert: nn.Module,
    cfg: LoRAConfig,
    freeze_base: bool = True,
) -> Dict[str, Any]:
    """
    Replace selected nn.Linear layers inside HuBERT with LoRALinear wrappers.

    Safe strategy:
      - Iterate parent modules via named_modules()
      - Replace only via parent.named_children() (no parent maps, no cycles)
      - Do NOT store parent refs inside nn.Module objects

    Returns info dict with counts.
    """
    if freeze_base:
        for p in hubert.parameters():
            p.requires_grad = False

    replaced = 0

    # Iterate over a snapshot of modules to avoid modifying while iterating
    for parent_name, parent in list(hubert.named_modules()):
        # Replace only direct children of this parent
        for child_name, child in list(parent.named_children()):
            if isinstance(child, nn.Linear) and _matches(child_name, cfg.target_keywords):
                setattr(parent, child_name, LoRALinear(child, r=cfg.r, alpha=cfg.alpha, dropout=cfg.dropout))
                replaced += 1

    # Make LoRA params trainable
    for m in hubert.modules():
        if isinstance(m, LoRALinear):
            m.lora_A.requires_grad = True
            m.lora_B.requires_grad = True
            # base already frozen inside LoRALinear

    trainable, total = _count_params(hubert)
    return {
        "replaced_linear": replaced,
        "trainable_params": trainable,
        "total_params": total,
        "trainable_ratio": float(trainable) / float(max(1, total)),
    }
