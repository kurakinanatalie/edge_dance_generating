from __future__ import annotations

from pathlib import Path
from typing import Optional

from audio_encoders.wavlm_cache import build_wavlm_cache
from edge_integration.edge_runner import run_edge_from_cache


def run_exp10_wavlm_lora(
    music_dir: Path,
    cache_dir: Path,
    projector_ckpt: Path,
    edge_repo_dir: Path,
    checkpoint: Path,
    render_dir: Path,
    motion_dir: Optional[Path] = None,
    device: str = "cuda",
    chunk_len: int = 150,
    max_tracks: Optional[int] = None,
    max_chunks_per_track: Optional[int] = None,
    out_length: float = 10.0,
    cfg_target: float = 7.5,
    no_render: bool = True,
) -> int:
    """
    Experiment 10:
    WavLM Base + trained Projector/LoRA checkpoint + EDGE inference.

    Note:
        This entrypoint assumes the WavLM-side training checkpoint already exists.
        It is designed for reproducible inference/evaluation from saved weights.
    """
    total_chunks = build_wavlm_cache(
        music_dir=Path(music_dir),
        cache_dir=Path(cache_dir),
        device=device,
        projector_ckpt=Path(projector_ckpt),
        chunk_len=chunk_len,
        max_tracks=max_tracks,
        max_chunks_per_track=max_chunks_per_track,
    )

    run_edge_from_cache(
        edge_repo_dir=Path(edge_repo_dir),
        feature_cache_dir=Path(cache_dir),
        music_dir=Path(music_dir),
        checkpoint=Path(checkpoint),
        render_dir=Path(render_dir),
        out_length=out_length,
        save_motions=(motion_dir is not None),
        motion_save_dir=Path(motion_dir) if motion_dir is not None else None,
        no_render=no_render,
        cfg_target=cfg_target,
    )

    return int(total_chunks)
