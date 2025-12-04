import numpy as np

def time_resample(x, new_len: int):
    """
    Uniformly resample along the time axis to match EDGE's 30 FPS.
    Uses linear interpolation frame-wise (fast & sufficient for 50→30 Hz).

    x: np.ndarray of shape (T, D)
    new_len: desired length along time axis
    """
    T, D = x.shape
    if T == new_len:
        return x
    t_old = np.linspace(0.0, 1.0, T)
    t_new = np.linspace(0.0, 1.0, new_len)
    out = np.empty((new_len, D), dtype=np.float32)
    for d in range(D):
        out[:, d] = np.interp(t_new, t_old, x[:, d])
    return out

def to_chunks(arr_2d: np.ndarray, chunk_len: int = 150):
    """
    Split a (T, D) array into consecutive non-overlapping chunks of length 'chunk_len'.
    Discard tail if shorter than chunk_len to keep shapes consistent.
    """
    T = arr_2d.shape[0]
    n = T // chunk_len
    return [
        arr_2d[i*chunk_len:(i+1)*chunk_len]
        for i in range(n)
        if arr_2d[i*chunk_len:(i+1)*chunk_len].shape[0] == chunk_len
    ]