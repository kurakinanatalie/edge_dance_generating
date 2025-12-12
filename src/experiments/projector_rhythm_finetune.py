# src/experiments/projector_rhythm_finetune.py

"""
Rhythm-aware self-supervised fine-tuning of the Projector module.

Goal:
    Adapt the Projector on top of frozen HuBERT features, without relying
    on Jukebox features or any heavy external target. Instead, we:
      - extract a simple rhythm envelope from audio (onset strength)
      - resample it to the same frame rate as the Projector outputs
      - encourage the Projector features to be:
            * smooth in time where rhythm is weak
            * allowed to change more where rhythm is strong
      - regularise the feature variance to avoid trivial collapse.

This is inspired by the idea of rhythm-aware representations from
music-to-dance works like Danceba ("Align Your Rhythm"), but implemented
in a very lightweight and Colab-friendly way.

HuBERT is always frozen here. Only the Projector is trained.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoFeatureExtractor, AutoModel

from audio_encoders.hubert_utils import time_resample, to_chunks
from models.projector import Projector


HUBERT_NAME = "facebook/hubert-base-ls960"


# ---------------------------------------------------------------------------
# 1. HuBERT + rhythm envelope extraction
# ---------------------------------------------------------------------------

def load_hubert(device: str = "cuda"):
    """
    Load HuBERT feature extractor and encoder on the given device.

    We use:
      - AutoFeatureExtractor: handles normalization & framing of raw audio
      - AutoModel: the HuBERT encoder itself (frozen in this script)
    """
    fe = AutoFeatureExtractor.from_pretrained(HUBERT_NAME)
    model = AutoModel.from_pretrained(HUBERT_NAME).to(device)
    model.eval()
    return fe, model


@torch.no_grad()
def extract_hubert_frames(
    wav_path: Path,
    feature_extractor,
    hubert_model,
    device: str = "cuda",
    target_sr: int = 16000,
) -> np.ndarray:
    """
    Read audio, convert to mono, resample to 16 kHz, and run HuBERT.

    Returns:
        np.ndarray of shape (T_h, 768), where T_h is the number of HuBERT frames
        (approximately ~50 Hz).
    """
    import librosa

    y, sr = sf.read(str(wav_path), always_2d=False)
    if y.ndim == 2:
        y = y.mean(axis=-1)  # stereo -> mono

    if sr != target_sr:
        y = librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    inputs = feature_extractor(y, sampling_rate=sr, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    out = hubert_model(**inputs)
    hidden = out.last_hidden_state.squeeze(0).detach().cpu().numpy().astype(np.float32)
    return hidden  # (T_h, 768)


def extract_rhythm_envelope(
    wav_path: Path,
    target_len: int,
    target_sr: int = 16000,
) -> np.ndarray:
    """
    Compute a simple rhythm/onset envelope and resample it to `target_len`.

    We use librosa.onset.onset_strength as a lightweight approximation of
    the rhythm intensity over time, then linearly resample it to match the
    number of frames used by the Projector (~30 Hz).

    Returns:
        np.ndarray of shape (target_len,), values roughly in [0, 1].
    """
    import librosa

    y, sr = sf.read(str(wav_path), always_2d=False)
    if y.ndim == 2:
        y = y.mean(axis=-1)

    if sr != target_sr:
        y = librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    # Onset strength envelope (librosa handles STFT/hops internally)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)  # (T_onset,)

    # Normalise per clip
    onset_env = onset_env.astype(np.float32)
    if onset_env.size == 0:
        onset_env = np.zeros((1,), dtype=np.float32)

    onset_env -= onset_env.min()
    onset_env /= (onset_env.max() + 1e-6)

    # Resample to target_len via linear interpolation
    t_old = np.linspace(0.0, 1.0, num=onset_env.shape[0])
    t_new = np.linspace(0.0, 1.0, num=target_len)
    env_resampled = np.interp(t_new, t_old, onset_env).astype(np.float32)

    return env_resampled  # (target_len,)


def prepare_rhythm_aware_chunks(
    music_dir: Path,
    device: str = "cuda",
    max_tracks: Optional[int] = None,
    max_chunks_per_track: Optional[int] = None,
    chunk_len: int = 150,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build inputs for rhythm-aware Projector training.

    For each .wav file:
      - HuBERT frames: (T_h,768) -> resampled to ~30 Hz -> z-score normalised
      - rhythm envelope: onset_strength -> resampled to same length (~30 Hz)
      - both sequences are split into non-overlapping chunks of length `chunk_len`

    Returns:
        X: tensor of shape [N, chunk_len, 768]
        R: tensor of shape [N, chunk_len]   (values in [0,1], rhythm intensity)
    """
    music_dir = Path(music_dir)
    wavs = sorted(p for p in music_dir.glob("*.wav") if p.is_file())
    if max_tracks is not None:
        wavs = wavs[:max_tracks]

    if not wavs:
        raise RuntimeError(f"No .wav files found in {music_dir}")

    fe, hubert_model = load_hubert(device=device)

    xs = []
    rs = []

    for w in wavs:
        try:
            # HuBERT features (T_h,768)
            hub = extract_hubert_frames(w, fe, hubert_model, device=device)

            # 50 Hz -> 30 Hz (approx)
            T30 = max(int(round(hub.shape[0] * 30.0 / 50.0)), 1)
            z = time_resample(hub, T30)  # (T_30,768)

            # per-clip z-score
            z = (z - z.mean(0, keepdims=True)) / (z.std(0, keepdims=True) + 1e-6)

            # rhythm envelope resampled to T30
            env = extract_rhythm_envelope(w, target_len=T30)  # (T_30,)

            # split into chunks
            feat_chunks = to_chunks(z, chunk_len=chunk_len)
            env_chunks = to_chunks(env[:, None], chunk_len=chunk_len)  # make (T,1) to reuse

            if max_chunks_per_track is not None:
                feat_chunks = feat_chunks[:max_chunks_per_track]
                env_chunks = env_chunks[:max_chunks_per_track]

            if not feat_chunks or not env_chunks:
                print(f"[data] too few frames in {w.name}: feats {z.shape}, env {env.shape}")
                continue

            assert len(feat_chunks) == len(env_chunks), "feature/env chunk mismatch"

            for fch, ech in zip(feat_chunks, env_chunks):
                # fch: (chunk_len,768), ech: (chunk_len,1)
                xs.append(torch.from_numpy(fch.astype(np.float32)))
                rs.append(torch.from_numpy(ech.squeeze(-1).astype(np.float32)))
        except Exception as e:
            print(f"[data] error processing {w.name}: {e}")

    if not xs:
        raise RuntimeError("No valid chunks for rhythm-aware training.")

    X = torch.stack(xs, dim=0)  # [N, chunk_len, 768]
    R = torch.stack(rs, dim=0)  # [N, chunk_len]

    print(f"[data] Prepared X shape {X.shape}, R shape {R.shape}")
    return X, R


# ---------------------------------------------------------------------------
# 2. Rhythm-aware loss for Projector outputs
# ---------------------------------------------------------------------------

def rhythm_aware_loss(
    z: torch.Tensor,
    r: torch.Tensor,
    lambda_smooth: float = 1.0,
    lambda_var: float = 0.1,
) -> torch.Tensor:
    """
    Compute a rhythm-aware self-supervised loss for Projector outputs.

    Args:
        z: Projector outputs, shape [B, T, D]
        r: rhythm envelope, shape [B, T], values roughly in [0,1]
        lambda_smooth: weight for the smoothness term
        lambda_var: weight for the variance regularisation

    Components:
        1) Smoothness with rhythm gating:
            - For each time step t > 0, consider delta_t = z_t - z_{t-1}.
            - Use a per-step weight w_t = 1 - r_t.
              -> where rhythm is strong (r_t near 1), we allow larger changes
                 (small penalty).
              -> where rhythm is weak (r_t near 0), we penalise changes more.

        2) Variance regularisation:
            - We encourage the per-dimension variance of z (over batch+time)
              to be close to 1.0 to avoid trivial collapse or exploding scales.
    """
    # Ensure shapes
    assert z.dim() == 3, "z must have shape [B,T,D]"
    assert r.dim() == 2, "r must have shape [B,T]"
    B, T, D = z.shape

    if T < 2:
        raise ValueError("Need at least 2 time steps for smoothness.")

    # 1) Smoothness with rhythm-dependent weighting
    dz = z[:, 1:, :] - z[:, :-1, :]  # [B,T-1,D]
    # use r at time t (corresponding to z[:,1:,:])
    r_t = r[:, 1:]  # [B,T-1]
    w_t = 1.0 - r_t  # [B,T-1], low weight near strong rhythm

    # squared L2 per step
    sq = (dz ** 2).mean(dim=-1)  # [B,T-1]
    smooth = (w_t * sq).mean()

    # 2) Variance regularisation (over batch+time)
    z_flat = z.reshape(B * T, D)
    var = z_flat.var(dim=0)  # [D]
    var_target = torch.ones_like(var)
    var_loss = ((var - var_target) ** 2).mean()

    total = lambda_smooth * smooth + lambda_var * var_loss
    return total


# ---------------------------------------------------------------------------
# 3. Training loop for rhythm-aware Projector
# ---------------------------------------------------------------------------

def run_rhythm_projector_finetune(
    music_dir: Path,
    output_path: Path,
    device: str = "cuda",
    max_tracks: Optional[int] = None,
    max_chunks_per_track: Optional[int] = None,
    chunk_len: int = 150,
    epochs: int = 3,
    batch_size: int = 4,
    lr: float = 1e-4,
    lambda_smooth: float = 1.0,
    lambda_var: float = 0.1,
):
    """
    High-level entry point for rhythm-aware fine-tuning of the Projector.

    Args:
        music_dir: directory with .wav files (short music clips).
        output_path: where to save the fine-tuned Projector weights (.pt).
        device: "cuda" or "cpu".
        max_tracks: optionally limit number of tracks for small experiments.
        max_chunks_per_track: optionally limit number of chunks per track.
        chunk_len: temporal length of each training chunk (in frames).
        epochs: number of epochs to train.
        batch_size: batch size for training.
        lr: learning rate for AdamW.
        lambda_smooth: weight for rhythm-aware smoothness term.
        lambda_var: weight for variance regularisation term.

    Behaviour:
        - prepares (X,R) via prepare_rhythm_aware_chunks()
        - trains a new Projector with rhythm_aware_loss()
        - saves the fine-tuned state_dict() to output_path
    """
    use_cuda = torch.cuda.is_available() and device == "cuda"
    device = "cuda" if use_cuda else "cpu"
    print(f"[train] Using device: {device}")

    # 1) Data preparation
    X, R = prepare_rhythm_aware_chunks(
        music_dir=music_dir,
        device=device,
        max_tracks=max_tracks,
        max_chunks_per_track=max_chunks_per_track,
        chunk_len=chunk_len,
    )

    dataset = TensorDataset(X, R)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 2) Projector to train
    projector = Projector().to(device)
    projector.train()

    # 3) Optimiser
    optimizer = torch.optim.AdamW(projector.parameters(), lr=lr)

    global_step = 0
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        for batch_idx, (xb, rb) in enumerate(loader):
            xb = xb.to(device)  # [B,T,768]
            rb = rb.to(device)  # [B,T]

            optimizer.zero_grad()
            z = projector(xb)  # [B,T,4800]

            loss = rhythm_aware_loss(
                z, rb,
                lambda_smooth=lambda_smooth,
                lambda_var=lambda_var,
            )
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            global_step += 1

            if (batch_idx + 1) % 10 == 0:
                avg = running_loss / 10
                print(
                    f"[train] epoch {epoch} step {global_step} "
                    f"batch {batch_idx+1}/{len(loader)} "
                    f"loss {avg:.6f}"
                )
                running_loss = 0.0

        print(f"[train] End of epoch {epoch}/{epochs}")

    # 4) Save fine-tuned Projector
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(projector.state_dict(), str(output_path))
    print(f"[train] Saved rhythm-aware Projector to {output_path}")
