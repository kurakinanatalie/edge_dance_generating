from __future__ import annotations

from pathlib import Path
from typing import Optional

from audio_encoders.hubert_cache import build_hubert_cache
from edge_integration.edge_runner import run_edge_from_cache


def run_exp01_baseline(
    music_dir: Path,
    cache_dir: Path,
    edge_repo_dir: Path,
    checkpoint: Path,
    render_dir: Path,
    motion_dir: Optional[Path] = None,
    device: str = "cuda",
    chunk_len: int = 150,
    out_length: float = 10.0,
    cfg_target: float = 7.5,
    no_render: bool = True,
) -> int:
    """
    Experiment 01 baseline:
    frozen HuBERT + default Projector + EDGE inference from cached features.

    Returns:
        Number of cached chunks saved.
    """
    music_dir = Path(music_dir)
    cache_dir = Path(cache_dir)
    edge_repo_dir = Path(edge_repo_dir)
    checkpoint = Path(checkpoint)
    render_dir = Path(render_dir)
    if motion_dir is not None:
        motion_dir = Path(motion_dir)

    total_chunks = build_hubert_cache(
        music_dir=music_dir,
        cache_dir=cache_dir,
        device=device,
        chunk_len=chunk_len,
        projector_ckpt=None,
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
