
# src/experiments/exp04_hubert_top3_e2e.py

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple
import numpy as np

import torch
import torch.nn as nn
import soundfile as sf
import librosa
from transformers import AutoFeatureExtractor, AutoModel

from models.projector import Projector

MODEL_NAME = "facebook/hubert-base-ls960"


def load_audio_mono(path: Path, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    """Load wav, convert to mono, resample to target_sr."""
    y, sr = sf.read(str(path), always_2d=False)
    if y.ndim == 2:
        y = y.mean(axis=-1)
    if sr != target_sr:
        y = librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    return y.astype(np.float32), sr


def onset_strength_30fps(y: np.ndarray, sr: int, target_fps: int = 30, hop_length: int = 512) -> np.ndarray:
    """Compute onset strength and resample it to 30 FPS timeline."""
    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    t_old = librosa.frames_to_time(np.arange(len(onset)), sr=sr, hop_length=hop_length)
    if len(t_old) == 0:
        return np.zeros((1,), dtype=np.float32)

    T = int(np.ceil(t_old[-1] * target_fps)) + 1
    t_new = np.arange(T) / target_fps
    r = np.interp(t_new, t_old, onset).astype(np.float32)

    # Z-score normalization for stable loss scale
    r = (r - r.mean()) / (r.std() + 1e-6)
    return r


def freeze_all(module: torch.nn.Module) -> None:
    """Freeze all parameters."""
    for p in module.parameters():
        p.requires_grad = False


def unfreeze_hubert_topk(hubert: torch.nn.Module, top_k_layers: int = 3) -> Tuple[int, int]:
    """
    Freeze HuBERT, then unfreeze the top-k transformer layers.
    Returns (trainable_params, total_params).
    """
    freeze_all(hubert)

    layers = hubert.encoder.layers
    for layer in layers[-top_k_layers:]:
        for p in layer.parameters():
            p.requires_grad = True

    trainable = sum(p.numel() for p in hubert.parameters() if p.requires_grad)
    total = sum(p.numel() for p in hubert.parameters())
    return int(trainable), int(total)


def train_exp04_e2e(
    music_dir: Path,
    out_projector_ckpt: Path,
    out_hubert_ckpt: Path,
    device: str = "cuda",
    epochs: int = 2,
    chunk_len: int = 150,
    max_chunks_per_track: int = 3,
    lr_projector: float = 2e-4,
    lr_hubert: float = 5e-6,
    grad_accum_steps: int = 4,
    use_amp: bool = True,
    top_k_layers: int = 3,
    seed: int = 123,
) -> None:
    """
    Strict end-to-end fine-tuning:
      wav -> HuBERT -> resample to 30fps -> Projector -> energy
      loss = MSE(zscore(energy), zscore(onset_strength))

    Trainable:
      - Projector (all)
      - HuBERT top-k layers
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

    fe = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
    hubert = AutoModel.from_pretrained(MODEL_NAME).to(device)
    projector = Projector().to(device)

    trainable, total = unfreeze_hubert_topk(hubert, top_k_layers=top_k_layers)
    print(f"[exp04] HuBERT trainable: {trainable/1e6:.2f}M / {total/1e6:.2f}M (top {top_k_layers})")

    # Ensure projector is trainable
    for p in projector.parameters():
        p.requires_grad = True

    params = [
        {"params": [p for p in projector.parameters() if p.requires_grad], "lr": lr_projector},
        {"params": [p for p in hubert.parameters() if p.requires_grad], "lr": lr_hubert},
    ]
    opt = torch.optim.AdamW(params, weight_decay=1e-4)

    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and device.startswith("cuda")))
    loss_fn = nn.MSELoss()

    hubert.train()
    projector.train()
    opt.zero_grad(set_to_none=True)

    step = 0
    for ep in range(epochs):
        print(f"[exp04] Epoch {ep+1}/{epochs}")
        order = np.random.permutation(len(wavs)).tolist()

        for idx in order:
            wav_path = wavs[idx]
            try:
                y, sr = load_audio_mono(wav_path, target_sr=16000)
                r_30 = onset_strength_30fps(y, sr, target_fps=30)  # (T,)

                inputs = fe(y, sampling_rate=sr, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}

                with torch.cuda.amp.autocast(enabled=(use_amp and device.startswith("cuda"))):
                    out = hubert(**inputs)
                    h50 = out.last_hidden_state.squeeze(0)  # (T50,768)

                T50 = h50.shape[0]
                T30 = max(int(round(T50 * 30.0 / 50.0)), 1)

                # Resample features to 30 FPS using torch interpolation
                h50_t = h50.transpose(0, 1).unsqueeze(0)  # (1,768,T50)
                h30_t = torch.nn.functional.interpolate(h50_t, size=T30, mode="linear", align_corners=False)
                h30 = h30_t.squeeze(0).transpose(0, 1)    # (T30,768)

                L = min(h30.shape[0], len(r_30))
                if L < chunk_len:
                    continue

                h30 = h30[:L]
                r = torch.from_numpy(r_30[:L]).to(device=device, dtype=h30.dtype)

                # Per-clip normalization of features
                h30 = (h30 - h30.mean(dim=0, keepdim=True)) / (h30.std(dim=0, keepdim=True) + 1e-6)

                n_chunks = L // chunk_len
                take = min(n_chunks, max_chunks_per_track)
                if take <= 0:
                    continue
                starts = np.random.choice(n_chunks, size=take, replace=False)

                track_loss = None
                chunks_used = 0

                for sidx in starts:
                    start = int(sidx * chunk_len)
                    x = h30[start:start+chunk_len]     # (150,768)
                    rr = r[start:start+chunk_len]      # (150,)

                    with torch.cuda.amp.autocast(enabled=(use_amp and device.startswith("cuda"))):
                        yproj = projector(x)                      # (150,4800)
                        energy = torch.linalg.norm(yproj, dim=-1) # (150,)
                        energy = (energy - energy.mean()) / (energy.std() + 1e-6)
                        loss = loss_fn(energy, rr)

                    track_loss = loss if track_loss is None else (track_loss + loss)
                    chunks_used += 1

                if chunks_used == 0:
                    continue
    
                # Average loss across selected chunks
                track_loss = track_loss / float(chunks_used)

                scaler.scale(track_loss / grad_accum_steps).backward()

                if ((step + 1) % grad_accum_steps) == 0:
                    scaler.step(opt)
                    scaler.update()
                    opt.zero_grad(set_to_none=True)

                if (step % 10) == 0:
                    print(f"[exp04] step {step:5d} | {wav_path.stem} | loss {track_loss.item():.4f} | chunks {chunks_used}")
                
                step += 1

            except Exception as e:
                print(f"[exp04] Error {wav_path.name}: {e}")

        print(f"[exp04] End epoch {ep+1}/{epochs}")

    hubert.eval()
    projector.eval()
    torch.save(projector.state_dict(), str(out_projector_ckpt))
    torch.save(hubert.state_dict(), str(out_hubert_ckpt))

    print(f"[exp04] Saved projector to {out_projector_ckpt}")
    print(f"[exp04] Saved hubert    to {out_hubert_ckpt}")
