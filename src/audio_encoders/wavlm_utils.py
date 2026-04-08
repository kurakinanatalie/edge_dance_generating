from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import soundfile as sf
import torch
from transformers import AutoFeatureExtractor, AutoModel


WAVLM_BASE = "microsoft/wavlm-base"
WAVLM_BASE_PLUS = "microsoft/wavlm-base-plus"


def load_audio_mono(
    wav_path: Path,
    target_sr: int = 16000,
) -> Tuple[np.ndarray, int]:
    """
    Load audio, convert to mono, and resample to target_sr if needed.
    """
    import librosa

    y, sr = sf.read(str(wav_path), always_2d=False)
    if y.ndim == 2:
        y = y.mean(axis=-1)

    y = y.astype(np.float32)

    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    return y, sr


def load_wavlm(
    model_name: str = WAVLM_BASE,
    device: str = "cuda",
):
    """
    Load WavLM feature extractor and encoder.
    """
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    return feature_extractor, model


@torch.no_grad()
def extract_wavlm_frames(
    wav_path: Path,
    feature_extractor,
    wavlm_model,
    device: str = "cuda",
    target_sr: int = 16000,
) -> np.ndarray:
    """
    Extract WavLM hidden states from a .wav file.

    Returns:
        np.ndarray of shape (T, 768)
    """
    y, sr = load_audio_mono(wav_path, target_sr=target_sr)

    inputs = feature_extractor(
        y,
        sampling_rate=sr,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    out = wavlm_model(**inputs)
    hidden = out.last_hidden_state.squeeze(0).detach().cpu().numpy().astype(np.float32)
    return hidden
