from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
import json

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoFeatureExtractor, AutoModel

from audio_encoders.wavlm_utils import WAVLM_BASE, load_audio_mono
from audio_encoders.lora import LoRALinear, LoRAConfig
from models.projector import Projector


TARGET_SR = 16000
TARGET_FPS = 30


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


def extract_rhythm_envelope_from_audio(
    y: np.ndarray,
    sr: int,
    target_len: int,
) -> np.ndarray:
    """
    Compute onset strength and resample to target_len.
    """
    import librosa

    onset_env = librosa.onset.onset_strength(y=y, sr=sr).astype(np.float32)

    if onset_env.size == 0:
        onset_env = np.zeros((1,), dtype=np.float32)

    onset_env -= onset_env.min()
    onset_env /= (onset_env.max() + 1e-6)

    t_old = np.linspace(0.0, 1.0, num=onset_env.shape[0])
    t_new = np.linspace(0.0, 1.0, num=target_len)
    out = np.interp(t_new, t_old, onset_env).astype(np.float32)
    return out


def prepare_wave_chunks(
    music_dir: Path,
    chunk_len: int = 150,
    target_sr: int = TARGET_SR,
    target_fps: int = TARGET_FPS,
    max_tracks: Optional[int] = None,
    max_chunks_per_track: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Prepare fixed-length waveform chunks and aligned rhythm envelopes.

    Returns:
        Xwav: [N, S]
        R:    [N, chunk_len]
    """
    music_dir = Path(music_dir)
    wavs = sorted(p for p in music_dir.glob("*.wav") if p.is_file())
    if max_tracks is not None:
        wavs = wavs[:max_tracks]

    if not wavs:
        raise RuntimeError(f"No .wav files found in {music_dir}")

    chunk_samples = int(round(chunk_len * target_sr / target_fps))

    xs: List[torch.Tensor] = []
    rs: List[torch.Tensor] = []

    for wav_path in wavs:
        try:
            y, sr = load_audio_mono(wav_path, target_sr=target_sr)
            total_chunks = len(y) // chunk_samples

            if total_chunks <= 0:
                print(f"[data] too short: {wav_path.name}")
                continue

            if max_chunks_per_track is not None:
                total_chunks = min(total_chunks, max_chunks_per_track)

            usable_samples = total_chunks * chunk_samples
            y = y[:usable_samples]

            env = extract_rhythm_envelope_from_audio(
                y=y,
                sr=sr,
                target_len=total_chunks * chunk_len,
            )

            for i in range(total_chunks):
                s0 = i * chunk_samples
                s1 = s0 + chunk_samples
                r0 = i * chunk_len
                r1 = r0 + chunk_len

                wav_chunk = y[s0:s1]
                env_chunk = env[r0:r1]

                if wav_chunk.shape[0] != chunk_samples:
                    continue
                if env_chunk.shape[0] != chunk_len:
                    continue

                xs.append(torch.from_numpy(wav_chunk.astype(np.float32)))
                rs.append(torch.from_numpy(env_chunk.astype(np.float32)))

        except Exception as e:
            print(f"[data] error processing {wav_path.name}: {e}")

    if not xs:
        raise RuntimeError("No valid waveform chunks prepared.")

    Xwav = torch.stack(xs, dim=0)
    R = torch.stack(rs, dim=0)
    print(f"[data] Prepared Xwav shape {Xwav.shape}, R shape {R.shape}")
    return Xwav, R


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

    info = {
        "replaced_linear": replaced,
        "trainable_params": trainable,
        "total_params": total,
        "trainable_ratio": trainable / max(total, 1),
    }
    return info


def save_lora_state_dict(
    model: nn.Module,
    output_path: Path,
    lora_cfg: LoRAConfig,
    model_name: str,
) -> None:
    """
    Save only LoRA parameters and metadata.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lora_state = {}
    for module_name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            lora_state[f"{module_name}.lora_A"] = module.lora_A.detach().cpu()
            lora_state[f"{module_name}.lora_B"] = module.lora_B.detach().cpu()

    payload = {
        "model_name": model_name,
        "lora_cfg": asdict(lora_cfg),
        "state_dict": lora_state,
    }
    torch.save(payload, str(output_path))
    print(f"[save] Saved WavLM LoRA weights to {output_path}")


def temporal_resample_torch(
    x: torch.Tensor,
    target_len: int,
) -> torch.Tensor:
    """
    Resample [B, T, D] to [B, target_len, D].
    """
    x = x.transpose(1, 2)
    x = F.interpolate(x, size=target_len, mode="linear", align_corners=False)
    x = x.transpose(1, 2)
    return x


def normalize_per_clip(
    x: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Per-clip z-score normalization over time.
    """
    mean = x.mean(dim=1, keepdim=True)
    std = x.std(dim=1, keepdim=True)
    return (x - mean) / (std + eps)


def rhythm_aware_loss(
    z: torch.Tensor,
    r: torch.Tensor,
    lambda_var: float = 0.1,
) -> torch.Tensor:
    """
    Rhythm-aware self-supervised loss.

    z: [B, T, D]
    r: [B, T]
    """
    if z.shape[1] < 2:
        raise ValueError("Need at least 2 frames for rhythm-aware loss.")

    dz = z[:, 1:, :] - z[:, :-1, :]
    r_t = r[:, 1:]
    w_t = 1.0 - r_t

    sq = (dz ** 2).mean(dim=-1)
    smooth = (w_t * sq).mean()

    z_flat = z.reshape(-1, z.shape[-1])
    var = z_flat.var(dim=0)
    var_target = torch.ones_like(var)
    var_loss = ((var - var_target) ** 2).mean()

    return smooth + lambda_var * var_loss


def smoothness_loss(
    z: torch.Tensor,
    order: int = 2,
) -> torch.Tensor:
    """
    Optional temporal smoothness regularization on projected features.
    """
    if z.shape[1] <= order:
        return torch.zeros((), device=z.device, dtype=z.dtype)

    d = z
    for _ in range(order):
        d = d[:, 1:, :] - d[:, :-1, :]
    return (d ** 2).mean()


def build_model_inputs(
    xb: torch.Tensor,
    feature_extractor,
    device: str,
):
    """
    Convert waveform batch [B, S] into WavLM model inputs.
    """
    batch_np = [row.detach().cpu().numpy() for row in xb]
    inputs = feature_extractor(
        batch_np,
        sampling_rate=TARGET_SR,
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    return inputs


def train_wavlm_lora_e2e(
    music_dir: Path,
    out_projector_ckpt: Path,
    out_wavlm_lora_ckpt: Path,
    model_name: str = WAVLM_BASE,
    lora_cfg: Optional[LoRAConfig] = None,
    device: str = "cuda",
    max_tracks: Optional[int] = None,
    max_chunks_per_track: Optional[int] = None,
    chunk_len: int = 150,
    epochs: int = 2,
    batch_size: int = 2,
    lr_projector: float = 2e-4,
    lr_lora: float = 1e-4,
    grad_accum_steps: int = 4,
    use_amp: bool = True,
    lambda_var: float = 0.1,
    smooth_lambda: float = 0.0,
    smooth_order: int = 2,
    seed: int = 123,
    meta_json_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    End-to-end rhythm-aware training of WavLM + LoRA + Projector.

    The training signal is self-supervised:
    - rhythm-aware temporal gating
    - optional smoothness regularization
    """
    if lora_cfg is None:
        lora_cfg = LoRAConfig(
            r=8,
            alpha=16.0,
            dropout=0.05,
            target_keywords=(
                "q_proj",
                "k_proj",
                "v_proj",
                "out_proj",
                "intermediate_dense",
                "output_dense",
            ),
        )

    torch.manual_seed(seed)
    np.random.seed(seed)

    use_cuda = torch.cuda.is_available() and device.startswith("cuda")
    device = "cuda" if use_cuda else "cpu"
    print(f"[wavlm] Using MODEL_NAME = {model_name}")
    print(f"[wavlm] Using device = {device}")

    Xwav, R = prepare_wave_chunks(
        music_dir=Path(music_dir),
        chunk_len=chunk_len,
        max_tracks=max_tracks,
        max_chunks_per_track=max_chunks_per_track,
    )
    dataset = TensorDataset(Xwav, R)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    feature_extractor, wavlm_model = load_wavlm(model_name=model_name, device=device)
    lora_info = inject_lora_into_model(wavlm_model, lora_cfg=lora_cfg)

    projector = Projector().to(device)
    projector.train()
    wavlm_model.train()

    projector_params = [p for p in projector.parameters() if p.requires_grad]
    lora_params = [p for p in wavlm_model.parameters() if p.requires_grad]

    print(f"[wavlm] LoRA injection: {lora_info}")
    print(f"[wavlm] Projector trainable params: {sum(p.numel() for p in projector_params) / 1e6:.2f}M")
    print(f"[wavlm] WavLM trainable params (LoRA): {sum(p.numel() for p in lora_params) / 1e6:.2f}M")

    optimizer = torch.optim.AdamW(
        [
            {"params": projector_params, "lr": lr_projector},
            {"params": lora_params, "lr": lr_lora},
        ]
    )

    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and device.startswith("cuda")))

    global_step = 0
    epoch_logs: List[Dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        print(f"[wavlm] Epoch {epoch}/{epochs}")

        running_total = 0.0
        running_main = 0.0
        running_smooth = 0.0
        seen = 0

        optimizer.zero_grad(set_to_none=True)

        for batch_idx, (xb, rb) in enumerate(loader, start=1):
            xb = xb.to(device)
            rb = rb.to(device)

            inputs = build_model_inputs(xb, feature_extractor, device=device)

            with torch.amp.autocast("cuda", enabled=(use_amp and device.startswith("cuda"))):
                out = wavlm_model(**inputs)
                hidden = out.last_hidden_state
                hidden = temporal_resample_torch(hidden, target_len=chunk_len)
                hidden = normalize_per_clip(hidden)

                z = projector(hidden)

                loss_main = rhythm_aware_loss(
                    z,
                    rb,
                    lambda_var=lambda_var,
                )

                loss_s = torch.zeros((), device=device, dtype=z.dtype)
                if smooth_lambda > 0.0:
                    loss_s = smoothness_loss(z, order=smooth_order)

                loss = loss_main + smooth_lambda * loss_s
                loss_scaled = loss / grad_accum_steps

            scaler.scale(loss_scaled).backward()

            if batch_idx % grad_accum_steps == 0 or batch_idx == len(loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            running_total += float(loss.detach().item())
            running_main += float(loss_main.detach().item())
            running_smooth += float(loss_s.detach().item())
            seen += 1
            global_step += 1

            if batch_idx % 10 == 0 or batch_idx == len(loader):
                print(
                    f"[wavlm] epoch {epoch} batch {batch_idx}/{len(loader)} "
                    f"loss {running_total / max(seen,1):.6f} "
                    f"main {running_main / max(seen,1):.6f} "
                    f"smooth {running_smooth / max(seen,1):.6f}"
                )

        epoch_log = {
            "epoch": epoch,
            "loss": running_total / max(seen, 1),
            "main_loss": running_main / max(seen, 1),
            "smooth_loss": running_smooth / max(seen, 1),
        }
        epoch_logs.append(epoch_log)

    out_projector_ckpt = Path(out_projector_ckpt)
    out_projector_ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(projector.state_dict(), str(out_projector_ckpt))
    print(f"[save] Saved Projector weights to {out_projector_ckpt}")

    save_lora_state_dict(
        wavlm_model,
        output_path=Path(out_wavlm_lora_ckpt),
        lora_cfg=lora_cfg,
        model_name=model_name,
    )

    meta = {
        "model_name": model_name,
        "chunk_len": chunk_len,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr_projector": lr_projector,
        "lr_lora": lr_lora,
        "grad_accum_steps": grad_accum_steps,
        "lambda_var": lambda_var,
        "smooth_lambda": smooth_lambda,
        "smooth_order": smooth_order,
        "seed": seed,
        "max_tracks": max_tracks,
        "max_chunks_per_track": max_chunks_per_track,
        "lora_cfg": asdict(lora_cfg),
        "lora_info": lora_info,
        "epoch_logs": epoch_logs,
        "out_projector_ckpt": str(out_projector_ckpt),
        "out_wavlm_lora_ckpt": str(out_wavlm_lora_ckpt),
    }

    if meta_json_path is not None:
        meta_json_path = Path(meta_json_path)
        meta_json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(meta_json_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        print(f"[save] Saved meta JSON to {meta_json_path}")

    return meta
