# src/audio_encoders/hubert_cache.py

from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import soundfile as sf
import librosa
from transformers import AutoFeatureExtractor, AutoModel

from models.projector import Projector
from audio_encoders.hubert_utils import time_resample, to_chunks


MODEL_NAME = "facebook/hubert-base-ls960"


def load_hubert(device="cuda"):
    """
    Load HuBERT feature extractor and model in eval mode.
    Returns (feature_extractor, model).
    """
    fe = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device).eval()
    return fe, model


def extract_hubert(path: Path, fe, model, device="cuda", target_sr=16000):
    """
    Read audio, resample to 16 kHz, convert to mono, and run HuBERT encoder.
    Returns a float32 numpy array of shape (T, 768) at ~50 Hz.
    """
    y, sr = sf.read(str(path), always_2d=False)
    if y.ndim == 2:
        y = y.mean(axis=-1)  # convert to mono

    if sr != target_sr:
        y = librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    inputs = fe(y, sampling_rate=sr, return_tensors="pt")

    with torch.no_grad():
        out = model(**{k: v.to(device) for k, v in inputs.items()})

    return out.last_hidden_state.squeeze(0).cpu().numpy().astype(np.float32)


def build_hubert_cache(
    music_dir: Path,
    cache_dir: Path,
    device: str = "cuda",
    chunk_len: int = 150,
):
    """
    Build HuBERT-based feature cache compatible with EDGE (150 x 4800 slices).
    Saves .npy chunks to cache_dir/<song>/<i>.npy.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Load HuBERT + projector
    fe, hubert = load_hubert(device)
    projector = Projector().to(device).eval()

    wavs = sorted(music_dir.glob("*.wav"))
    if not wavs:
        raise RuntimeError(f"No .wav files found in {music_dir}")

    total_chunks = 0

    for w in wavs:
        try:
            # Step 1 — HuBERT embeddings
            hub = extract_hubert(w, fe, hubert, device=device)  # (T_hubert, 768)

            # Step 2 — Resample HuBERT timeline to ~30 FPS
            T30 = max(int(round(hub.shape[0] * 30.0 / 50.0)), 1)
            hub_30fps = time_resample(hub, T30)

            # Step 3 — Per-clip normalization
            hub_norm = (hub_30fps - hub_30fps.mean(0, keepdims=True)) / (
                hub_30fps.std(0, keepdims=True) + 1e-6
            )

            # Step 4 — Project to 4800-dim Jukebox-like space
            with torch.no_grad():
                proj_out = projector(torch.from_numpy(hub_norm).to(device))
                proj_np = proj_out.cpu().numpy().astype(np.float32)

            # Step 5 — Split into slices
            chunks = to_chunks(proj_np, chunk_len=chunk_len)
            if not chunks:
                print(f"Skipping {w.name}: too few frames ({proj_np.shape})")
                continue

            # Step 6 — Save chunks
            song_dir = cache_dir / w.stem
            song_dir.mkdir(parents=True, exist_ok=True)

            for i, ch in enumerate(chunks):
                np.save(song_dir / f"{i}.npy", ch)
                total_chunks += 1

            print(f"OK {w.name}: {len(chunks)} chunks saved → {song_dir}")

        except Exception as e:
            print(f"Error processing {w.name}: {e}")

    print(f"\nTotal chunks: {total_chunks}")
    return total_chunks