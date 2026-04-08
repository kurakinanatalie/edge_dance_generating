from __future__ import annotations

from pathlib import Path
from typing import Optional

from audio_encoders.wavlm_cache import build_wavlm_cache
from audio_encoders.wavlm_utils import WAVLM_BASE_PLUS
from edge_integration.edge_runner import run_edge_from_cache


def run_exp10b_wavlm_baseplus(
    music_dir: Path,
    cache_dir: Path,
    projector_ckpt: Path,
    edge_repo_dir: Path,
    checkpoint: Path,
    render_dir: Path,
    motion_dir: Optional[Path] = None,
    wavlm_lora_ckpt: Optional[Path] = None,
    device: str = "cuda",
    chunk_len: int = 150,
    max_tracks: Optional[int] = None,
    max_chunks_per_track: Optional[int] = None,
    out_length: float = 10.0,
    cfg_target: float = 7.5,
    no_render: bool = True,
) -> int:
    """
    Experiment 10b:
    WavLM Base+ + trained Projector checkpoint + EDGE inference.
    """
    total_chunks = build_wavlm_cache(
        music_dir=Path(music_dir),
        cache_dir=Path(cache_dir),
        device=device,
        model_name=WAVLM_BASE_PLUS,
        projector_ckpt=Path(projector_ckpt),
        wavlm_lora_ckpt=Path(wavlm_lora_ckpt) if wavlm_lora_ckpt is not None else None,
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
