from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Dict, Any

from audio_encoders.lora import LoRAConfig
from experiments.exp06_hubert_lora_e2e import train_exp06_lora_e2e


def run_exp09_hubert_lora_lambda_sweep(
    music_dir: Path,
    out_dir: Path,
    lambdas: Iterable[float] = (0.0, 0.01, 0.03, 0.05, 0.1),
    device: str = "cuda",
    epochs: int = 2,
    chunk_len: int = 150,
    max_chunks_per_track: int = 3,
    lr_projector: float = 2e-4,
    lr_lora: float = 1e-4,
    grad_accum_steps: int = 4,
    use_amp: bool = True,
    smooth_order: int = 2,
    seed: int = 123,
) -> List[Dict[str, Any]]:
    """
    Experiment 09:
    HuBERT + LoRA + smoothness lambda sweep.

    Returns:
        List of run metadata dictionaries.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

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

    rows: List[Dict[str, Any]] = []

    for lam in lambdas:
        lam_str = str(lam).replace(".", "p")
        run_dir = out_dir / f"lambda_{lam_str}"
        ckpt_dir = run_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        projector_ckpt = ckpt_dir / "projector.pt"
        hubert_ckpt = ckpt_dir / "hubert_lora.pt"

        train_exp06_lora_e2e(
            music_dir=Path(music_dir),
            out_projector_ckpt=projector_ckpt,
            out_hubert_ckpt=hubert_ckpt,
            lora_cfg=cfg,
            device=device,
            epochs=epochs,
            chunk_len=chunk_len,
            max_chunks_per_track=max_chunks_per_track,
            lr_projector=lr_projector,
            lr_lora=lr_lora,
            grad_accum_steps=grad_accum_steps,
            use_amp=use_amp,
            onset_smooth_window=None,
            smooth_lambda=float(lam),
            smooth_order=smooth_order,
            seed=seed,
        )

        rows.append(
            {
                "lambda": float(lam),
                "run_dir": str(run_dir),
                "projector_ckpt": str(projector_ckpt),
                "hubert_ckpt": str(hubert_ckpt),
            }
        )

    return rows
