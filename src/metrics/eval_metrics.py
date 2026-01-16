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

def summarize_run_from_csv(
    alignment_csv: Path,
    jerk_csv: Path,
    tag: str,
) -> Dict[str, Any]:
    """
    Reads already computed CSVs and returns mean alignment corr and mean jerk.
    alignment_csv must contain column f"{tag}_corr"
    jerk_csv must contain column f"{tag}_jerk"
    """
    a = pd.read_csv(alignment_csv)
    j = pd.read_csv(jerk_csv)

    out: Dict[str, Any] = {"tag": tag}

    corr_col = f"{tag}_corr"
    jerk_col = f"{tag}_jerk"

    out["Alignment_mean_corr"] = float(a[corr_col].mean()) if corr_col in a.columns and len(a) > 0 else float("nan")
    out["Mean_Jerk"] = float(j[jerk_col].mean()) if jerk_col in j.columns and len(j) > 0 else float("nan")
    out["n_tracks_alignment"] = int(len(a))
    out["n_tracks_jerk"] = int(len(j))
    return out


def summarize_runs_table(
    runs: List[Dict[str, Any]],
    base_dir: Path,
    out_csv: Path,
) -> pd.DataFrame:
    """
    Build a single summary table from a list of dict configs.

    runs format example:
    [
      {
        "name": "exp01 Baseline",
        "tag": "exp01",
        "alignment_csv": base_dir / "metrics_alignment_exp01_vs_exp06b.csv",
        "jerk_csv": base_dir / "metrics_jerk_exp01_vs_exp06b.csv",
      },
      ...
    ]

    This assumes each CSV contains columns for the given tag (tag_corr / tag_jerk).
    """
    rows: List[Dict[str, Any]] = []
    for r in runs:
        s = summarize_run_from_csv(
            alignment_csv=Path(r["alignment_csv"]),
            jerk_csv=Path(r["jerk_csv"]),
            tag=str(r["tag"]),
        )
        rows.append(
            {
                "Experiment": r.get("name", r["tag"]),
                "Alignment_mean_corr": s["Alignment_mean_corr"],
                "Mean_Jerk": s["Mean_Jerk"],
                "n_tracks_alignment": s["n_tracks_alignment"],
                "n_tracks_jerk": s["n_tracks_jerk"],
            }
        )

    df = pd.DataFrame(rows)
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print("Saved:", out_csv)
    return df
