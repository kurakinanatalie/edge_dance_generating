from __future__ import annotations

from pathlib import Path

from audio_encoders.lora import LoRAConfig
from experiments.exp06_hubert_lora_e2e import train_exp06_lora_e2e


def run_exp07_hubert_lora_onset_smooth(
    music_dir: Path,
    out_projector_ckpt: Path,
    out_hubert_ckpt: Path,
    device: str = "cuda",
    epochs: int = 2,
    chunk_len: int = 150,
    max_chunks_per_track: int = 3,
    lr_projector: float = 2e-4,
    lr_lora: float = 1e-4,
    grad_accum_steps: int = 4,
    use_amp: bool = True,
    onset_smooth_window: int = 5,
    seed: int = 123,
) -> None:
    """
    Experiment 07:
    HuBERT + LoRA on attention and FFN layers + onset smoothing.
    """
    cfg = LoRAConfig(
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

    train_exp06_lora_e2e(
        music_dir=Path(music_dir),
        out_projector_ckpt=Path(out_projector_ckpt),
        out_hubert_ckpt=Path(out_hubert_ckpt),
        lora_cfg=cfg,
        device=device,
        epochs=epochs,
        chunk_len=chunk_len,
        max_chunks_per_track=max_chunks_per_track,
        lr_projector=lr_projector,
        lr_lora=lr_lora,
        grad_accum_steps=grad_accum_steps,
        use_amp=use_amp,
        onset_smooth_window=onset_smooth_window,
        smooth_lambda=0.0,
        smooth_order=2,
        seed=seed,
    )
