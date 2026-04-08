from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch

from audio_encoders.wavlm_utils import (
    WAVLM_BASE,
    load_wavlm,
    extract_wavlm_frames,
)
from audio_encoders.hubert_utils import time_resample, to_chunks
from models.projector import Projector


def build_wavlm_cache(
    music_dir: Path,
    cache_dir: Path,
    device: str = "cuda",
    model_name: str = WAVLM_BASE,
    projector_ckpt: Optional[Path] = None,
    chunk_len: int = 150,
    max_tracks: Optional[int] = None,
    max_chunks_per_track: Optional[int] = None,
) -> int:
    """
    Build WavLM feature cache compatible with EDGE.

    Steps:
    - load wav files
    - extract WavLM features (~50 Hz)
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
            )  # (T,768)

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
