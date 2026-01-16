from __future__ import annotations

from pathlib import Path
from typing import Optional, Callable, Dict, Any

import numpy as np
import torch
import torch.nn.functional as F
import soundfile as sf
import librosa
from transformers import AutoFeatureExtractor, AutoModel

from models.projector import Projector

MODEL_NAME_DEFAULT = "facebook/hubert-base-ls960"


def _load_audio_mono(path: Path, target_sr: int = 16000) -> tuple[np.ndarray, int]:
    y, sr = sf.read(str(path), always_2d=False)
    if y.ndim == 2:
        y = y.mean(axis=-1)
    if sr != target_sr:
        y = librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    return y.astype(np.float32), sr


@torch.no_grad()
def build_feature_cache(
    music_dir: Path,
    cache_dir: Path,
    projector_ckpt: Path,
    hubert_ckpt: Optional[Path] = None,
    device: str = "cuda",
    chunk_len: int = 150,
    model_name: str = MODEL_NAME_DEFAULT,
    lora_inject_fn: Optional[Callable[[torch.nn.Module], Dict[str, Any]]] = None,
    per_clip_norm: bool = True,
) -> int:
    """
    Build EDGE conditioning cache:
      wav -> HuBERT -> resample 50Hz->30fps -> Projector -> (T,4800) -> chunk into (150,4800) .npy files

    If hubert_ckpt is provided, it is loaded with strict=False (to allow LoRA keys).
    If lora_inject_fn is provided, it is called on the hubert model BEFORE loading hubert_ckpt.
    """
    music_dir = Path(music_dir)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    wavs = sorted(music_dir.glob("*.wav"))
    if not wavs:
        raise RuntimeError(f"No wav files in {music_dir}")

    fe = AutoFeatureExtractor.from_pretrained(model_name)

    hubert = AutoModel.from_pretrained(model_name).to(device)
    hubert.eval()

    if lora_inject_fn is not None:
        info = lora_inject_fn(hubert)
        print("[cache] LoRA inject:", info)

    if hubert_ckpt is not None:
        sd_h = torch.load(str(hubert_ckpt), map_location=device)
        hubert.load_state_dict(sd_h, strict=False)
        hubert.eval()

    projector = Projector().to(device)
    sd_p = torch.load(str(projector_ckpt), map_location=device)
    projector.load_state_dict(sd_p, strict=True)
    projector.eval()

    saved = 0
    for wav_path in wavs:
        y, sr = _load_audio_mono(wav_path, target_sr=16000)

        inputs = fe(y, sampling_rate=sr, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        out = hubert(**inputs)
        h50 = out.last_hidden_state.squeeze(0)  # (T50,768)

        T50 = int(h50.shape[0])
        T30 = max(int(round(T50 * 30.0 / 50.0)), 1)

        h50_t = h50.transpose(0, 1).unsqueeze(0)  # (1,768,T50)
        h30_t = F.interpolate(h50_t, size=T30, mode="linear", align_corners=False)
        h30 = h30_t.squeeze(0).transpose(0, 1)    # (T30,768)

        if per_clip_norm:
            h30 = (h30 - h30.mean(dim=0, keepdim=True)) / (h30.std(dim=0, keepdim=True) + 1e-6)

        yproj = projector(h30)  # (T30,4800)
        yproj = yproj.detach().cpu().numpy().astype(np.float32)

        track_dir = cache_dir / wav_path.stem
        track_dir.mkdir(parents=True, exist_ok=True)

        n_chunks = yproj.shape[0] // chunk_len
        for i in range(n_chunks):
            chunk = yproj[i * chunk_len : (i + 1) * chunk_len]
            if chunk.shape == (chunk_len, 4800):
                np.save(track_dir / f"{i}.npy", chunk)
                saved += 1

    return int(saved)