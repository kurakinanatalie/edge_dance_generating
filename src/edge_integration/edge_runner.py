from pathlib import Path
import glob
import os
import torch
import types


def safe_patch_torch_load():
    """
    Patch torch.load to always use weights_only=False to support older checkpoints.
    """
    orig_load = torch.load

    def _load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return orig_load(*args, **kwargs)

    torch.load = _load


def adjust_cfg_scale(model, cfg_target: float):
    """
    Try to adjust CFG / guidance scale on model or model.diffusion if such attributes exist.
    """
    cand = []
    diff = getattr(model, "diffusion", None)
    if diff is not None:
        for name in ("cfg_scale", "scale", "guidance_scale", "guidance_weight"):
            if hasattr(diff, name):
                cand.append(("diffusion." + name, diff, name))

    for name in ("cfg_scale", "scale", "guidance_scale", "guidance_weight"):
        if hasattr(model, name):
            cand.append(("model." + name, model, name))

    changed = False
    for path, obj, name in cand:
        try:
            old = getattr(obj, name)
            if isinstance(old, (int, float)):
                setattr(obj, name, cfg_target)
                print(f"[CFG] {path}: {old} -> {cfg_target}")
                changed = True
        except Exception as e:
            print(f"[CFG] failed for {path}: {e}")

    if not changed:
        print("[CFG] no CFG/guidance scale attribute found (this is ok).")


def load_cached_batches(feature_cache_dir: Path, music_dir: Path, slice_len: int = 150, sample_size: int = 1):
    """
    Load (150,4800) chunks from feature_cache_dir/<song>/<i>.npy.
    Returns: list of tensors [K,150,4800] and list of filename lists.
    """
    import numpy as np

    feature_cache_dir = Path(feature_cache_dir)
    music_dir = Path(music_dir)

    dirs = sorted([d for d in feature_cache_dir.glob("*/") if d.is_dir()])
    music_wavs = sorted(music_dir.glob("*.wav"))
    default_wav = str(music_wavs[0]) if music_wavs else "placeholder.wav"

    batches = []
    names = []

    for d in dirs:
        npys = sorted(d.glob("*.npy"))
        if not npys:
            continue

        cond_list = []
        for path in npys:
            arr = np.load(path)
            if arr.ndim == 2 and arr.shape == (slice_len, 4800):
                cond_list.append(arr)

        if not cond_list:
            continue

        cond_sel = cond_list[:sample_size]
        import torch as _torch

        batches.append(_torch.from_numpy(np.stack(cond_sel, axis=0)))  # [K,150,4800]
        names.append([default_wav] * len(cond_sel))

    return batches, names


def run_edge_from_cache(
    edge_repo_dir: Path,
    feature_cache_dir: Path,
    music_dir: Path,
    checkpoint: Path,
    render_dir: Path,
    out_length: float = 10.0,
    save_motions: bool = False,
    motion_save_dir: Path | None = None,
    no_render: bool = False,
    cfg_target: float | None = None,
):
    """
    Minimal EDGE runner that uses cached (150x4800) features instead of Jukebox.
    """
    import sys

    edge_repo_dir = Path(edge_repo_dir)
    feature_cache_dir = Path(feature_cache_dir)
    music_dir = Path(music_dir)
    render_dir = Path(render_dir)
    checkpoint = str(checkpoint)

    os.chdir(edge_repo_dir)
    sys.path.append(str(edge_repo_dir))

    import torch  # re-import under patched load
    safe_patch_torch_load()

    from EDGE import EDGE  # type: ignore

    # Prepare a small "opt" namespace compatible with original test.py
    opt = types.SimpleNamespace()
    opt.feature_type = "jukebox"
    opt.out_length = float(out_length)
    opt.render_dir = str(render_dir)
    opt.checkpoint = checkpoint
    opt.music_dir = str(music_dir)
    opt.save_motions = bool(save_motions)
    opt.motion_save_dir = str(motion_save_dir) if motion_save_dir else "eval/motions"
    opt.no_render = bool(no_render)
    opt.use_cached_features = True
    opt.feature_cache_dir = str(feature_cache_dir)

    model = EDGE(opt.feature_type, opt.checkpoint)
    model.eval()

    if cfg_target is not None:
        adjust_cfg_scale(model, cfg_target)

    slice_len = 150
    sample_size = 1
    batches, names = load_cached_batches(feature_cache_dir, music_dir, slice_len, sample_size)

    if not batches:
        print("[runner] No valid chunks (expected .npy with shape (150,4800)).")
        return

    print(f"[runner] tracks found: {len(batches)}")
    fk_out = opt.motion_save_dir if opt.save_motions else None

    for i in range(len(batches)):
        data_tuple = (None, batches[i], names[i])
        model.render_sample(
            data_tuple,
            "test",
            opt.render_dir,
            render_count=-1,
            fk_out=fk_out,
            render=not opt.no_render,
        )

    print("Done.")