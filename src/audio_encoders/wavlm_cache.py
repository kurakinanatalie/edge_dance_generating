from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import torch
import torch.nn as nn

from audio_encoders.wavlm_utils import (
    WAVLM_BASE,
    load_wavlm,
    extract_wavlm_frames,
)
from audio_encoders.hubert_utils import time_resample, to_chunks
from audio_encoders.lora import LoRALinear, LoRAConfig
from models.projector import Projector


def inject_lora_into_model(
    model: nn.Module,
    lora_cfg: LoRAConfig,
) -> Dict[str, Any]:
    """
    Freeze the base model and replace selected Linear layers with LoRALinear.
    """
    for p in model.parameters():
        p.requires_grad = False

    replaced = 0

    def _inject(module: nn.Module) -> None:
        nonlocal replaced
        for child_name, child in list(module.named_children()):
            if isinstance(child, nn.Linear) and any(
                key in child_name for key in lora_cfg.target_keywords
            ):
                setattr(
                    module,
                    child_name,
                    LoRALinear(
                        base=child,
                        r=lora_cfg.r,
                        alpha=lora_cfg.alpha,
                        dropout=lora_cfg.dropout,
                    ),
                )
                replaced += 1
            else:
                _inject(child)

    _inject(model)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    return {
        "replaced_linear": replaced,
        "trainable_params": trainable,
        "total_params": total,
        "trainable_ratio": trainable / max(total, 1),
    }


def load_lora_state_dict(
    model: nn.Module,
    lora_ckpt_path: Path,
    device: str = "cuda",
) -> Dict[str, Any]:
    """
    Load LoRA payload saved by train_wavlm_lora_e2e.py.
    """
    payload = torch.load(str(lora_ckpt_path), map_location=device)
    lora_cfg = LoRAConfig(**payload["lora_cfg"])
    info = inject_lora_into_model(model, lora_cfg=lora_cfg)

    state_dict = payload["state_dict"]
    missing = []
    for module_name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            key_a = f"{module_name}.lora_A"
            key_b = f"{module_name}.lora_B"
            if key_a in state_dict and key_b in state_dict:
                module.lora_A.data.copy_(state_dict[key_a].to(device))
                module.lora_B.data.copy_(state_dict[key_b].to(device))
            else:
                missing.append(module_name)

    if missing:
        print(f"[wavlm_cache] Warning: missing LoRA weights for {len(missing)} modules")

    return {
        "model_name": payload.get("model_name"),
        "lora_cfg": payload.get("lora_cfg"),
        "inject_info": info,
    }


def build_wavlm_cache(
    music_dir: Path,
    cache_dir: Path,
    device: str = "cuda",
    model_name: str = WAVLM_BASE,
    projector_ckpt: Optional[Path] = None,
    wavlm_lora_ckpt: Optional[Path] = None,
    chunk_len: int = 150,
    max_tracks: Optional[int] = None,
    max_chunks_per_track: Optional[int] = None,
) -> int:
    """
    Build WavLM feature cache compatible with EDGE.

    Steps:
    - load wav files
    - extract WavLM features
    - optionally load LoRA weights
    - resample to ~30 Hz
    - normalize per clip
    - project 768 -> 4800
    - split into chunks and save as .npy

    Returns:
        Number of chunks saved.
    """
    music_dir = Path(music_dir)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    wavs = sorted(p for p in music_dir.glob("*.wav") if p.is_file())
    if max_tracks is not None:
        wavs = wavs[:max_tracks]

    if not wavs:
        raise RuntimeError(f"No .wav files found in {music_dir}")

    feature_extractor, wavlm_model = load_wavlm(model_name=model_name, device=device)

    if wavlm_lora_ckpt is not None:
        info = load_lora_state_dict(
            wavlm_model,
            lora_ckpt_path=Path(wavlm_lora_ckpt),
            device=device,
        )
        print(f"[wavlm_cache] Loaded LoRA: {info}")

    wavlm_model.eval()

    projector = Projector().to(device)
    projector.eval()

    if projector_ckpt is not None:
        state = torch.load(str(projector_ckpt), map_location=device)
        projector.load_state_dict(state, strict=True)

    total_chunks = 0

    for wav_path in wavs:
        try:
            frames = extract_wavlm_frames(
                wav_path,
                feature_extractor=feature_extractor,
                wavlm_model=wavlm_model,
                device=device,
            )

            t30 = max(int(round(frames.shape[0] * 30.0 / 50.0)), 1)
            z = time_resample(frames, t30)

            z = (z - z.mean(0, keepdims=True)) / (z.std(0, keepdims=True) + 1e-6)

            x = torch.from_numpy(z.astype(np.float32)).unsqueeze(0).to(device)
            with torch.no_grad():
                proj = projector(x).squeeze(0).detach().cpu().numpy().astype(np.float32)

            chunks = to_chunks(proj, chunk_len=chunk_len)
            if max_chunks_per_track is not None:
                chunks = chunks[:max_chunks_per_track]

            stem = wav_path.stem
            for i, chunk in enumerate(chunks):
                out_path = cache_dir / f"{stem}_chunk{i:03d}.npy"
                np.save(out_path, chunk)
                total_chunks += 1

        except Exception as e:
            print(f"[wavlm_cache] Error processing {wav_path.name}: {e}")

    print(f"[wavlm_cache] Saved {total_chunks} chunks to {cache_dir}")
    return total_chunks
