from __future__ import annotations

from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List

import numpy as np
import pandas as pd
import pickle
import soundfile as sf
import librosa


def zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return (x - x.mean()) / (x.std() + 1e-8)


def load_motion_energy(pkl_path: Path, use_trans: bool = True) -> Tuple[np.ndarray, int]:
    with open(pkl_path, "rb") as f:
        d = pickle.load(f)
    poses = np.asarray(d["smpl_poses"], dtype=np.float32)
    trans = np.asarray(d["smpl_trans"], dtype=np.float32)

    x = np.concatenate([poses, trans], axis=1) if use_trans else poses
    v = np.linalg.norm(np.diff(x, axis=0), axis=1)
    v = np.concatenate([[v[0]], v])
    return v, 30


def load_audio_onset(
    wav_path: Path,
    target_sr: int = 22050,
    target_fps: int = 30,
    hop_length: int = 512,
) -> np.ndarray:
    y, sr = sf.read(str(wav_path), always_2d=False)
    if y.ndim == 2:
        y = y.mean(axis=-1)
    if sr != target_sr:
        y = librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    t_old = librosa.frames_to_time(np.arange(len(onset)), sr=sr, hop_length=hop_length)
    if len(t_old) == 0:
        return np.zeros((1,), dtype=np.float32)

    T = int(np.ceil(t_old[-1] * target_fps)) + 1
    t_new = np.arange(T) / target_fps
    return np.interp(t_new, t_old, onset).astype(np.float32)


def best_xcorr(a: np.ndarray, b: np.ndarray, max_lag_frames: int = 15) -> Tuple[float, int]:
    a = zscore(a)
    b = zscore(b)
    L = min(len(a), len(b))
    a = a[:L]
    b = b[:L]

    best_corr = -1e9
    best_lag = 0

    for lag in range(-max_lag_frames, max_lag_frames + 1):
        if lag < 0:
            aa = a[-lag:]
            bb = b[:len(aa)]
        elif lag > 0:
            aa = a[:-lag]
            bb = b[lag : lag + len(aa)]
        else:
            aa, bb = a, b

        if len(aa) < 20:
            continue

        c = float(np.corrcoef(aa, bb)[0, 1])
        if c > best_corr:
            best_corr = c
            best_lag = lag

    return best_corr, best_lag


def wav_for_stem(stem: str, music_dir: Path) -> Optional[Path]:
    s = stem.rstrip("_")
    cand = music_dir / f"{s}.wav"
    if cand.exists():
        return cand

    hits = list(music_dir.glob(f"{s}*.wav"))
    return hits[0] if hits else None


def alignment_table(
    runA: Path,
    runB: Path,
    tagA: str,
    tagB: str,
    out_csv: Path,
    music_dir: Path,
    use_trans: bool = True,
    max_lag_frames: int = 15,
    onset_target_sr: int = 22050,
    onset_hop_length: int = 512,
) -> pd.DataFrame:
    A = {p.stem: p for p in (Path(runA) / "motions").rglob("*.pkl")}
    B = {p.stem: p for p in (Path(runB) / "motions").rglob("*.pkl")}
    common = sorted(set(A.keys()) & set(B.keys()))
    print("Common PKLs:", len(common))

    rows: List[Dict[str, Any]] = []
    for stem in common:
        wav = wav_for_stem(stem, music_dir=music_dir)
        if wav is None:
            continue

        aud = load_audio_onset(
            wav,
            target_sr=onset_target_sr,
            target_fps=30,
            hop_length=onset_hop_length,
        )
        mA, _ = load_motion_energy(A[stem], use_trans=use_trans)
        mB, _ = load_motion_energy(B[stem], use_trans=use_trans)

        cA, lA = best_xcorr(aud, mA, max_lag_frames=max_lag_frames)
        cB, lB = best_xcorr(aud, mB, max_lag_frames=max_lag_frames)

        rows.append(
            {
                "track": stem,
                f"{tagA}_corr": cA,
                f"{tagA}_lag": lA,
                f"{tagB}_corr": cB,
                f"{tagB}_lag": lB,
            }
        )

    df = pd.DataFrame(rows).sort_values("track")
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    print("Saved:", out_csv)
    if len(df) > 0:
        print("Mean", tagA, "corr:", df[f"{tagA}_corr"].mean())
        print("Mean", tagB, "corr:", df[f"{tagB}_corr"].mean())
    return df


def load_poses(pkl_path: Path) -> np.ndarray:
    with open(pkl_path, "rb") as f:
        d = pickle.load(f)
    return np.asarray(d["smpl_poses"], dtype=np.float32)


def mean_jerk(poses: np.ndarray) -> float:
    if poses.shape[0] < 4:
        return float("nan")
    v = np.diff(poses, axis=0)
    a = np.diff(v, axis=0)
    j = np.diff(a, axis=0)
    return float(np.linalg.norm(j, axis=1).mean())


def jerk_table(
    runA: Path,
    runB: Path,
    tagA: str,
    tagB: str,
    out_csv: Path,
) -> pd.DataFrame:
    A = {p.stem: p for p in (Path(runA) / "motions").rglob("*.pkl")}
    B = {p.stem: p for p in (Path(runB) / "motions").rglob("*.pkl")}
    common = sorted(set(A.keys()) & set(B.keys()))
    print("Common PKLs:", len(common))

    rows: List[Dict[str, Any]] = []
    for stem in common:
        jA = mean_jerk(load_poses(A[stem]))
        jB = mean_jerk(load_poses(B[stem]))
        rows.append({"track": stem, f"{tagA}_jerk": jA, f"{tagB}_jerk": jB})

    df = pd.DataFrame(rows).sort_values("track")
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    print("Saved:", out_csv)
    if len(df) > 0:
        print("Mean", tagA, "jerk:", df[f"{tagA}_jerk"].mean())
        print("Mean", tagB, "jerk:", df[f"{tagB}_jerk"].mean())
    return df