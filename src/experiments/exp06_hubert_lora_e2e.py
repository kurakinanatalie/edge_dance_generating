# src/experiments/exp06_hubert_lora_e2e.py

from __future__ import annotations

from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import soundfile as sf
import librosa
from transformers import AutoFeatureExtractor, AutoModel

from models.projector import Projector
from audio_encoders.lora import LoRAConfig, inject_lora_into_hubert

MODEL_NAME = "facebook/hubert-base-ls960"


def _load_audio_mono(path: Path, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    """Load wav, convert to mono, resample to target_sr."""
    y, sr = sf.read(str(path), always_2d=False)
    if y.ndim == 2:
        y = y.mean(axis=-1)
    if sr != target_sr:
        y = librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    return y.astype(np.float32), sr


def _onset_strength_30fps(
    y: np.ndarray,
    sr: int,
    target_fps: int = 30,
    hop_length: int = 512,
) -> np.ndarray:
    """Compute onset strength and resample it to 30 FPS timeline."""
    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    t_old = librosa.frames_to_time(np.arange(len(onset)), sr=sr, hop_length=hop_length)
    if len(t_old) == 0:
        return np.zeros((1,), dtype=np.float32)

    T = int(np.ceil(t_old[-1] * target_fps)) + 1
    t_new = np.arange(T) / target_fps
    r = np.interp(t_new, t_old, onset).astype(np.float32)

    # z-score for stable loss scale
    r = (r - r.mean()) / (r.std() + 1e-6)
    return r


def _zscore_torch(x: torch.Tensor) -> torch.Tensor:
    return (x - x.mean()) / (x.std() + 1e-6)


def train_exp06_lora_e2e(
    music_dir: Path,
    out_projector_ckpt: Path,
    out_hubert_ckpt: Path,
    lora_cfg: LoRAConfig,
    device: str = "cuda",
    epochs: int = 2,
    chunk_len: int = 150,
    max_chunks_per_track: int = 3,
    lr_projector: float = 2e-4,
    lr_lora: float = 1e-4,
    grad_accum_steps: int = 4,
    use_amp: bool = True,
    seed: int = 123,
) -> None:
    """
    Exp06: strict end-to-end wav -> HuBERT(+LoRA) -> Projector -> energy
    loss = MSE(zscore(energy), zscore(onset_strength))

    Trainable:
      - Projector (all params)
      - HuBERT LoRA params only (base frozen inside inject_lora_into_hubert)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    music_dir = Path(music_dir)
    wavs = sorted(music_dir.glob("*.wav"))
    if not wavs:
        raise RuntimeError(f"No .wav files found in {music_dir}")

    out_projector_ckpt = Path(out_projector_ckpt)
    out_hubert_ckpt = Path(out_hubert_ckpt)
    out_projector_ckpt.parent.mkdir(parents=True, exist_ok=True)
    out_hubert_ckpt.parent.mkdir(parents=True, exist_ok=True)

    # Models
    fe = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
    hubert = AutoModel.from_pretrained(MODEL_NAME).to(device)
    projector = Projector().to(device)

    # Inject LoRA and freeze base weights
    info = inject_lora_into_hubert(hubert, cfg=lora_cfg, freeze_base=True)
    print("[exp06] LoRA injection:", info)

    # Ensure projector trainable
    for p in projector.parameters():
        p.requires_grad = True

    # Build optimizer param groups: projector + LoRA-trainable params
    lora_params = [p for p in hubert.parameters() if p.requires_grad]
    proj_params = [p for p in projector.parameters() if p.requires_grad]

    print(f"[exp06] Projector trainable params: {sum(p.numel() for p in proj_params)/1e6:.2f}M")
    print(f"[exp06] HuBERT trainable params (LoRA): {sum(p.numel() for p in lora_params)/1e6:.2f}M")

    params = [
        {"params": proj_params, "lr": lr_projector},
        {"params": lora_params, "lr": lr_lora},
    ]
    opt = torch.optim.AdamW(params, weight_decay=1e-4)

    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and device.startswith("cuda")))
    loss_fn = nn.MSELoss()

    hubert.train()
    projector.train()
    opt.zero_grad(set_to_none=True)

    step = 0
    for ep in range(epochs):
        print(f"[exp06] Epoch {ep+1}/{epochs}")
        order = np.random.permutation(len(wavs)).tolist()

        for idx in order:
            wav_path = wavs[idx]
            try:
                y, sr = _load_audio_mono(wav_path, target_sr=16000)
                r_30_np = _onset_strength_30fps(y, sr, target_fps=30)  # (T,)

                inputs = fe(y, sampling_rate=sr, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}

                # Forward HuBERT (keep graph for LoRA)
                with torch.cuda.amp.autocast(enabled=(use_amp and device.startswith("cuda"))):
                    out = hubert(**inputs)
                    h50 = out.last_hidden_state.squeeze(0)  # (T50,768)

                T50 = h50.shape[0]
                T30 = max(int(round(T50 * 30.0 / 50.0)), 1)

                # Resample HuBERT features to 30 FPS (torch, keeps grad)
                h50_t = h50.transpose(0, 1).unsqueeze(0)  # (1,768,T50)
                h30_t = F.interpolate(h50_t, size=T30, mode="linear", align_corners=False)
                h30 = h30_t.squeeze(0).transpose(0, 1)    # (T30,768)

                L = min(h30.shape[0], len(r_30_np))
                if L < chunk_len:
                    continue

                h30 = h30[:L]
                r = torch.from_numpy(r_30_np[:L]).to(device=device, dtype=h30.dtype)

                # Per-clip normalization of features (stable)
                h30 = (h30 - h30.mean(dim=0, keepdim=True)) / (h30.std(dim=0, keepdim=True) + 1e-6)

                n_chunks = L // chunk_len
                take = min(n_chunks, max_chunks_per_track)
                if take <= 0:
                    continue

                starts = np.random.choice(n_chunks, size=take, replace=False)

                # accumulate loss over selected chunks (single backward per track)
                track_loss = 0.0
                for sidx in starts:
                    start = int(sidx * chunk_len)
                    x = h30[start:start+chunk_len]   # (150,768)
                    rr = r[start:start+chunk_len]    # (150,)

                    with torch.cuda.amp.autocast(enabled=(use_amp and device.startswith("cuda"))):
                        yproj = projector(x)                      # (150,4800)
                        energy = torch.linalg.norm(yproj, dim=-1) # (150,)
                        energy = _zscore_torch(energy)
                        loss = loss_fn(energy, rr)

                    track_loss = track_loss + loss

                track_loss = track_loss / float(len(starts))

                scaler.scale(track_loss / grad_accum_steps).backward()

                if ((step + 1) % grad_accum_steps) == 0:
                    scaler.step(opt)
                    scaler.update()
                    opt.zero_grad(set_to_none=True)

                if (step % 10) == 0:
                    print(f"[exp06] step {step:5d} | {wav_path.stem} | loss {track_loss.item():.4f} | chunks {len(starts)}")

                step += 1

            except Exception as e:
                print(f"[exp06] Error {wav_path.name}: {e}")

        print(f"[exp06] End epoch {ep+1}/{epochs}")

    hubert.eval()
    projector.eval()

    torch.save(projector.state_dict(), str(out_projector_ckpt))
    torch.save(hubert.state_dict(), str(out_hubert_ckpt))

    print(f"[exp06] Saved projector to {out_projector_ckpt}")
    print(f"[exp06] Saved hubert    to {out_hubert_ckpt}")
