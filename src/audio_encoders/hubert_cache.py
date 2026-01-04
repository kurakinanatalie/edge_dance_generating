# src/audio_encoders/hubert_cache.py

from pathlib import Path
from typing import Optional
import numpy as np
import torch
import soundfile as sf
import librosa
from transformers import AutoFeatureExtractor, AutoModel

from models.projector import Projector
from audio_encoders.hubert_utils import time_resample, to_chunks

MODEL_NAME = "facebook/hubert-base-ls960"


def load_hubert(device: str = "cuda", hubert_ckpt: Optional[Path] = None):
    """
    Load HuBERT feature extractor and model in eval mode.
    If hubert_ckpt is provided, load weights into the model.
    Returns (feature_extractor, model).
    """
    fe = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)

    if hubert_ckpt is not None:
        hubert_ckpt = Path(hubert_ckpt)
        print(f"[hubert_cache] Loading HuBERT weights from {hubert_ckpt}")
        state = torch.load(str(hubert_ckpt), map_location=device)
        model.load_state_dict(state, strict=True)

    model.eval()
    return fe, model


def extract_hubert(path: Path, fe, model, device: str = "cuda", target_sr: int = 16000):
    """
    Read audio, resample to 16 kHz, convert to mono, and run HuBERT encoder.
    Returns a float32 numpy array of shape (T, 768) at ~50 Hz.
    """
    y, sr = sf.read(str(path), always_2d=False)
    if y.ndim == 2:
        y = y.mean(axis=-1)

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
    projector_ckpt: Optional[Path] = None,
    hubert_ckpt: Optional[Path] = None,
) -> int:
    """
    Build HuBERT-based feature cache compatible with EDGE (chunk_len x 4800 slices).
    Saves .npy chunks to cache_dir/<song>/<i>.npy.

    If projector_ckpt is provided, loads Projector weights from that checkpoint.
    If hubert_ckpt is provided, loads HuBERT weights from that checkpoint.
    """
    music_dir = Path(music_dir)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Load HuBERT + projector
    fe, hubert = load_hubert(device=device, hubert_ckpt=hubert_ckpt)

    proj = Projector().to(device)
    if projector_ckpt is not None:
        projector_ckpt = Path(projector_ckpt)
        print(f"[hubert_cache] Loading Projector weights from {projector_ckpt}")
        state = torch.load(str(projector_ckpt), map_location=device)
        proj.load_state_dict(state, strict=True)
    proj.eval()

    wavs = sorted(music_dir.glob("*.wav"))
    if not wavs:
        raise RuntimeError(f"No .wav files found in {music_dir}")

    total_chunks = 0

    for w in wavs:
        try:
            # Step 1 — HuBERT embeddings (T_h,768) ~50Hz
            hub = extract_hubert(w, fe, hubert, device=device)

            # Step 2 — Resample to 30 FPS timeline
            T30 = max(int(round(hub.shape[0] * 30.0 / 50.0)), 1)
            hub_30 = time_resample(hub, T30)

            # Step 3 — Per-clip normalization
            hub_norm = (hub_30 - hub_30.mean(0, keepdims=True)) / (hub_30.std(0, keepdims=True) + 1e-6)

            # Step 4 — Project to 4800
            with torch.no_grad():
                proj_out = proj(torch.from_numpy(hub_norm).to(device))
                proj_np = proj_out.cpu().numpy().astype(np.float32)

            # Step 5 — Chunking
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
