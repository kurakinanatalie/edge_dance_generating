
# src/audio_encoders/lora.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple, List, Dict

import torch
import torch.nn as nn


@dataclass
class LoRAConfig:
    """
    Minimal LoRA config for Linear layers.

    r: LoRA rank
    alpha: scaling (effective scale = alpha / r)
    dropout: dropout applied to LoRA branch
    target_keywords: which module names to adapt (matched by substring)
    """
    r: int = 8
    alpha: float = 16.0
    dropout: float = 0.0
    target_keywords: Tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "out_proj")  # safe default


class LoRALinear(nn.Module):
    """
    Wraps an existing nn.Linear with a trainable low-rank update:
      y = W x + (alpha/r) * B(A(x))

    Base linear W is frozen by default (we freeze outside).
    """
    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError("LoRALinear expects nn.Linear as base")

        self.base = base
        self.in_features = base.in_features
        self.out_features = base.out_features

        self.r = int(r)
        self.alpha = float(alpha)
        self.scaling = self.alpha / max(self.r, 1)

        self.dropout = nn.Dropout(p=float(dropout)) if dropout and dropout > 0 else nn.Identity()

        # LoRA parameters
        # A: (r, in_features), B: (out_features, r)
        self.A = nn.Parameter(torch.zeros(self.r, self.in_features))
        self.B = nn.Parameter(torch.zeros(self.out_features, self.r))

        # Init: A ~ N(0, 0.02), B = 0 => start identical to base
        nn.init.normal_(self.A, mean=0.0, std=0.02)
        nn.init.zeros_(self.B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y0 = self.base(x)
        # x: (..., in_features)
        x_d = self.dropout(x)
        # (..., r)
        lora_h = torch.matmul(x_d, self.A.t())
        # (..., out_features)
        lora_y = torch.matmul(lora_h, self.B.t()) * self.scaling
        return y0 + lora_y


def _iter_named_modules(model: nn.Module) -> Iterable[Tuple[str, nn.Module]]:
    for name, module in model.named_modules():
        yield name, module


def _replace_module(parent: nn.Module, child_name: str, new_module: nn.Module) -> None:
    # child_name is attribute name within parent
    setattr(parent, child_name, new_module)


def inject_lora_into_hubert(
    hubert: nn.Module,
    cfg: Optional[LoRAConfig] = None,
    freeze_base: bool = True,
) -> Dict[str, int]:
    """
    Inject LoRA adapters into selected nn.Linear layers of a HuBERT model.

    Returns a dict with counts:
      { "replaced_linear": int, "trainable_params": int, "total_params": int }

    Notes:
    - We do NOT assume specific HF internal class names.
    - We match modules by name substring, so it's robust across versions.
    """
    if cfg is None:
        cfg = LoRAConfig()

    # Optionally freeze everything first
    if freeze_base:
        for p in hubert.parameters():
            p.requires_grad = False

    replaced = 0

    # We need parent pointers to replace children. Easiest: loop over parent.named_children recursively.
    # We'll do a manual traversal over named_modules, and for each module, inspect its direct children.
    for parent_name, parent in hubert.named_modules():
        for child_name, child in list(parent.named_children()):
            full_name = f"{parent_name}.{child_name}" if parent_name else child_name

            if isinstance(child, nn.Linear) and any(k in full_name for k in cfg.target_keywords):
                wrapped = LoRALinear(child, r=cfg.r, alpha=cfg.alpha, dropout=cfg.dropout)
                _replace_module(parent, child_name, wrapped)
                replaced += 1

    # Ensure LoRA params are trainable
    for m in hubert.modules():
        if isinstance(m, LoRALinear):
            m.A.requires_grad = True
            m.B.requires_grad = True

    trainable = sum(p.numel() for p in hubert.parameters() if p.requires_grad)
    total = sum(p.numel() for p in hubert.parameters())

    return {
        "replaced_linear": int(replaced),
        "trainable_params": int(trainable),
        "total_params": int(total),
    }


def lora_state_dict(hubert: nn.Module) -> Dict[str, torch.Tensor]:
    """
    Extract ONLY LoRA weights (A and B matrices) to save small checkpoints.
    """
    sd: Dict[str, torch.Tensor] = {}
    for name, module in hubert.named_modules():
        if isinstance(module, LoRALinear):
            sd[f"{name}.A"] = module.A.detach().cpu()
            sd[f"{name}.B"] = module.B.detach().cpu()
            sd[f"{name}.alpha"] = torch.tensor(module.alpha)
            sd[f"{name}.r"] = torch.tensor(module.r)
    return sd


def load_lora_state_dict(hubert: nn.Module, sd: Dict[str, torch.Tensor]) -> None:
    """
    Load LoRA weights into a model that already has LoRALinear injected.
    """
    for name, module in hubert.named_modules():
        if isinstance(module, LoRALinear):
            keyA = f"{name}.A"
            keyB = f"{name}.B"
            if keyA in sd and keyB in sd:
                module.A.data.copy_(sd[keyA].to(module.A.device))
                module.B.data.copy_(sd[keyB].to(module.B.device))
