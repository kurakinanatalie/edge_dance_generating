# src/audio_encoders/lora.py
# LoRA utilities (cycle-safe traversal: no named_modules recursion)

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Dict, Any, Iterable, List

import torch
import torch.nn as nn


@dataclass
class LoRAConfig:
    r: int = 8
    alpha: float = 16.0
    dropout: float = 0.05
    target_keywords: Tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "out_proj")


class LoRALinear(nn.Module):
    """
    Safe LoRA wrapper for nn.Linear:
      y = base(x) + scaling * ( (drop(x) @ A^T) @ B^T )
    """
    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError("LoRALinear expects an nn.Linear as base")

        self.base = base
        self.r = int(r)
        self.alpha = float(alpha)
        self.scaling = self.alpha / max(1, self.r)
        self.drop = nn.Dropout(p=float(dropout)) if dropout and dropout > 0 else nn.Identity()

        in_f = base.in_features
        out_f = base.out_features

        #Put LoRA params on same device/dtype as base
        dev = base.weight.device
        dt = base.weight.dtype

        self.lora_A = nn.Parameter(torch.zeros(self.r, in_f, device=dev, dtype=dt))
        self.lora_B = nn.Parameter(torch.zeros(out_f, self.r, device=dev, dtype=dt))

        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)

        # Freeze base weights by default
        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y0 = self.base(x)
        dx = self.drop(x)
        lora = (dx @ self.lora_A.t()) @ self.lora_B.t()
        return y0 + self.scaling * lora


def _matches(name: str, keywords: Iterable[str]) -> bool:
    name = name.lower()
    return any(k.lower() in name for k in keywords)


def _count_params(m: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in m.parameters())
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    return int(trainable), int(total)


def _iter_parents_bfs(root: nn.Module) -> List[Tuple[nn.Module, str, nn.Module]]:
    """
    Cycle-safe traversal:
    Returns list of triples: (parent_module, child_name, child_module)
    We use named_children() only and track visited module ids.
    """
    out = []
    queue = [root]
    visited = set()

    while queue:
        parent = queue.pop(0)
        pid = id(parent)
        if pid in visited:
            continue
        visited.add(pid)

        for child_name, child in parent.named_children():
            out.append((parent, child_name, child))
            # Continue BFS
            if isinstance(child, nn.Module):
                queue.append(child)

    return out


def inject_lora_into_hubert(
    hubert: nn.Module,
    cfg: LoRAConfig,
    freeze_base: bool = True,
) -> Dict[str, Any]:
    """
    Replace selected nn.Linear children with LoRALinear.
    This implementation is robust even if the module graph has accidental cycles.
    """
    if freeze_base:
        for p in hubert.parameters():
            p.requires_grad = False

    replaced = 0

    triples = _iter_parents_bfs(hubert)
    for parent, child_name, child in triples:
        if isinstance(child, nn.Linear) and _matches(child_name, cfg.target_keywords):
          lora_mod = LoRALinear(
            child,
            r=cfg.r,
            alpha=cfg.alpha,
            dropout=cfg.dropout,
        )

        #move LoRA module to same device
        lora_mod = lora_mod.to(child.weight.device)

        setattr(parent, child_name, lora_mod)
        replaced += 1

    # Ensure LoRA params are trainable
    for m in hubert.modules():
        if isinstance(m, LoRALinear):
            m.lora_A.requires_grad = True
            m.lora_B.requires_grad = True

    trainable, total = _count_params(hubert)
    return {
        "replaced_linear": replaced,
        "trainable_params": trainable,
        "total_params": total,
        "trainable_ratio": float(trainable) / float(max(1, total)),
    }
