from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable
import pickle

import numpy as np


def rot_x(deg: float) -> np.ndarray:
    rad = np.deg2rad(deg)
    c, s = np.cos(rad), np.sin(rad)
    return np.array([
        [1.0, 0.0, 0.0],
        [0.0, c, -s],
        [0.0, s, c],
    ], dtype=np.float32)


def rot_y(deg: float) -> np.ndarray:
    rad = np.deg2rad(deg)
    c, s = np.cos(rad), np.sin(rad)
    return np.array([
        [c, 0.0, s],
        [0.0, 1.0, 0.0],
        [-s, 0.0, c],
    ], dtype=np.float32)


def rot_z(deg: float) -> np.ndarray:
    rad = np.deg2rad(deg)
    c, s = np.cos(rad), np.sin(rad)
    return np.array([
        [c, -s, 0.0],
        [s, c, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)


def build_rotation_matrix(
    x_deg: float = 0.0,
    y_deg: float = 0.0,
    z_deg: float = 0.0,
) -> np.ndarray:
    """
    Combined rotation matrix.
    Applied as Rz @ Ry @ Rx.
    """
    rx = rot_x(x_deg)
    ry = rot_y(y_deg)
    rz = rot_z(z_deg)
    return rz @ ry @ rx


def rotate_xyz_array(arr: np.ndarray, R: np.ndarray) -> np.ndarray:
    """
    Rotate an array whose last dimension is 3.
    Supports:
    - (T, J, 3)
    - (J, 3)
    - (N, 3)
    - (T, J*3) -> reshaped internally if divisible by 3
    """
    arr = np.asarray(arr)

    if arr.ndim >= 1 and arr.shape[-1] == 3:
        flat = arr.reshape(-1, 3)
        out = flat @ R.T
        return out.reshape(arr.shape)

    if arr.ndim == 2 and arr.shape[1] % 3 == 0:
        old_shape = arr.shape
        reshaped = arr.reshape(arr.shape[0], arr.shape[1] // 3, 3)
        flat = reshaped.reshape(-1, 3)
        out = flat @ R.T
        out = out.reshape(reshaped.shape).reshape(old_shape)
        return out

    return arr


def maybe_rotate_object(obj: Any, R: np.ndarray) -> Any:
    """
    Recursively rotate any arrays that clearly represent xyz coordinates.

    This function is intentionally conservative:
    - it rotates numpy arrays with last dim = 3
    - it rotates 2D arrays with second dim divisible by 3
    - it recurses through dict/list/tuple
    - other objects are returned unchanged
    """
    if isinstance(obj, np.ndarray):
        return rotate_xyz_array(obj, R)

    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            out[k] = maybe_rotate_object(v, R)
        return out

    if isinstance(obj, list):
        return [maybe_rotate_object(v, R) for v in obj]

    if isinstance(obj, tuple):
        return tuple(maybe_rotate_object(v, R) for v in obj)

    return obj


def process_pickle_file(
    file_path: Path,
    x_deg: float = 180.0,
    y_deg: float = 0.0,
    z_deg: float = 0.0,
) -> None:
    file_path = Path(file_path)
    R = build_rotation_matrix(x_deg=x_deg, y_deg=y_deg, z_deg=z_deg)

    with open(file_path, "rb") as f:
        obj = pickle.load(f)

    obj = maybe_rotate_object(obj, R)

    with open(file_path, "wb") as f:
        pickle.dump(obj, f)

    print(f"[postprocess] Rotated pickle: {file_path}")


def process_npy_file(
    file_path: Path,
    x_deg: float = 180.0,
    y_deg: float = 0.0,
    z_deg: float = 0.0,
) -> None:
    file_path = Path(file_path)
    R = build_rotation_matrix(x_deg=x_deg, y_deg=y_deg, z_deg=z_deg)

    arr = np.load(file_path, allow_pickle=True)
    arr = maybe_rotate_object(arr, R)
    np.save(file_path, arr)

    print(f"[postprocess] Rotated npy: {file_path}")


def process_npz_file(
    file_path: Path,
    x_deg: float = 180.0,
    y_deg: float = 0.0,
    z_deg: float = 0.0,
) -> None:
    file_path = Path(file_path)
    R = build_rotation_matrix(x_deg=x_deg, y_deg=y_deg, z_deg=z_deg)

    data = np.load(file_path, allow_pickle=True)
    out = {}

    for k in data.files:
        out[k] = maybe_rotate_object(data[k], R)

    np.savez(file_path, **out)
    print(f"[postprocess] Rotated npz: {file_path}")


def process_motion_file(
    file_path: Path,
    x_deg: float = 180.0,
    y_deg: float = 0.0,
    z_deg: float = 0.0,
) -> None:
    file_path = Path(file_path)
    suffix = file_path.suffix.lower()

    if suffix == ".pkl":
        process_pickle_file(file_path, x_deg=x_deg, y_deg=y_deg, z_deg=z_deg)
    elif suffix == ".npy":
        process_npy_file(file_path, x_deg=x_deg, y_deg=y_deg, z_deg=z_deg)
    elif suffix == ".npz":
        process_npz_file(file_path, x_deg=x_deg, y_deg=y_deg, z_deg=z_deg)


def process_motion_dir(
    motion_dir: Path,
    x_deg: float = 180.0,
    y_deg: float = 0.0,
    z_deg: float = 0.0,
) -> None:
    """
    Rotate all supported motion files inside a directory.

    Default:
    x_deg = 180.0

    This is a practical first fix for an upside-down character.
    If later Blender needs Y-up -> Z-up conversion, we can adjust this.
    """
    motion_dir = Path(motion_dir)
    if not motion_dir.exists():
        print(f"[postprocess] Motion directory not found: {motion_dir}")
        return

    files = list(motion_dir.rglob("*.pkl")) + list(motion_dir.rglob("*.npy")) + list(motion_dir.rglob("*.npz"))
    if not files:
        print(f"[postprocess] No motion files found in: {motion_dir}")
        return

    for f in files:
        process_motion_file(f, x_deg=x_deg, y_deg=y_deg, z_deg=z_deg)
