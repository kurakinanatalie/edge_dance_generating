from __future__ import annotations

from pathlib import Path
from typing import Optional

from experiments.projector_rhythm_finetune import run_rhythm_projector_finetune
from audio_encoders.hubert_cache import build_hubert_cache
from edge_integration.edge_runner import run_edge_from_cache


def run_exp02_projector_rhythm(
    music_dir: Path,
    projector_ckpt_out: Path,
    cache_dir: Path,
    edge_repo_dir: Path,
    checkpoint: Path,
    render_dir: Path,
    motion_dir: Optional[Path] = None,
    device: str = "cuda",
    chunk_len: int = 150,
    epochs: int = 5,
    batch_size: int = 4,
    lr: float = 1e-4,
    lambda_smooth: float = 1.0,
    lambda_var: float = 0.1,
    max_tracks: Optional[int] = None,
    max_chunks_per_track: Optional[int] = None,
    out_length: float = 10.0,
    cfg_target: float = 7.5,
    no_render: bool = True,
) -> int:
    """
    Experiment 02:
    frozen HuBERT + rhythm-aware Projector fine-tuning + EDGE inference.

    Returns:
        Number of cached chunks saved.
    """
    music_dir = Path(music_dir)
    projector_ckpt_out = Path(projector_ckpt_out)
    cache_dir = Path(cache_dir)
    edge_repo_dir = Path(edge_repo_dir)
    checkpoint = Path(checkpoint)
    render_dir = Path(render_dir)
    if motion_dir is not None:
        motion_dir = Path(motion_dir)

    run_rhythm_projector_finetune(
        music_dir=music_dir,
        output_path=projector_ckpt_out,
        device=device,
        max_tracks=max_tracks,
        max_chunks_per_track=max_chunks_per_track,
        chunk_len=chunk_len,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        lambda_smooth=lambda_smooth,
        lambda_var=lambda_var,
    )

    total_chunks = build_hubert_cache(
        music_dir=music_dir,
        cache_dir=cache_dir,
        device=device,
        chunk_len=chunk_len,
        projector_ckpt=projector_ckpt_out,
        hubert_ckpt=None,
    )

    run_edge_from_cache(
        edge_repo_dir=edge_repo_dir,
        feature_cache_dir=cache_dir,
        music_dir=music_dir,
        checkpoint=checkpoint,
        render_dir=render_dir,
        out_length=out_length,
        save_motions=(motion_dir is not None),
        motion_save_dir=motion_dir,
        no_render=no_render,
        cfg_target=cfg_target,
    )

    return int(total_chunks)
