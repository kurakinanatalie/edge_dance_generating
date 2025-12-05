# src/edge_integration/edge_setup.py

import subprocess
import sys
from pathlib import Path


EDGE_REPO_URL = "https://github.com/Stanford-TML/EDGE.git"
EDGE_DEFAULT_DIR = Path("/content/EDGE")
CHECKPOINT_MIN_BYTES = 500_000_000  # sanity check: ~500MB


def run_cmd(cmd, cwd: Path | None = None):
    """
    Run a shell command and raise if it fails.
    """
    print("CMD:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(cwd) if cwd is not None else None)


def clone_edge_repo(edge_dir: Path = EDGE_DEFAULT_DIR):
    """
    Clone the official EDGE repository if it does not exist yet.
    """
    edge_dir = Path(edge_dir)
    if edge_dir.exists():
        print(f"[EDGE setup] Repo already exists at {edge_dir}")
        return edge_dir

    run_cmd(["git", "clone", EDGE_REPO_URL, str(edge_dir)])
    print(f"[EDGE setup] Cloned EDGE into {edge_dir}")
    return edge_dir


def ensure_checkpoint(edge_dir: Path = EDGE_DEFAULT_DIR):
    """
    Ensure that EDGE checkpoint exists and looks valid.

    1) Try the official download_model.sh script.
    2) If the checkpoint is missing or suspiciously small, download via gdown.
    3) Make sure there is checkpoint.pt at the repository root.
    """
    edge_dir = Path(edge_dir)
    model_dir = edge_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    ckpt_model = model_dir / "checkpoint.pt"
    ckpt_root = edge_dir / "checkpoint.pt"

    # Step 1: try official script
    try:
        print("[EDGE setup] Running download_model.sh ...")
        run_cmd(["bash", "download_model.sh"], cwd=edge_dir)
    except Exception as e:
        print("[EDGE setup] download_model.sh failed or not available:", e)

    # Step 2: check size, decide if we need gdown fallback
    need_gdown = True
    if ckpt_model.exists():
        size = ckpt_model.stat().st_size
        print(f"[EDGE setup] Found model/checkpoint.pt, size = {size} bytes")
        if size >= CHECKPOINT_MIN_BYTES:
            need_gdown = False
        else:
            print("[EDGE setup] Checkpoint too small, will re-download via gdown.")
    else:
        print("[EDGE setup] model/checkpoint.pt not found, will download via gdown.")

    if need_gdown:
        # Fallback via gdown, same ID as used previously in the notebook
        print("[EDGE setup] Installing gdown and downloading checkpoint via Google Drive ...")
        run_cmd([sys.executable, "-m", "pip", "install", "-q", "gdown"])
        run_cmd(
            [
                "gdown",
                "--id",
                "1BAR712cVEqB8GR37fcEihRV_xOC-fZrZ",
                "-O",
                str(ckpt_model),
            ]
        )
        size = ckpt_model.stat().st_size
        print(f"[EDGE setup] gdown download complete, size = {size} bytes")
        if size < CHECKPOINT_MIN_BYTES:
            raise RuntimeError(
                f"Downloaded checkpoint is still too small ({size} bytes); "
                "Google Drive may have returned an HTML warning page."
            )

    # Step 3: ensure root checkpoint.pt points to model/checkpoint.pt
    try:
        if ckpt_root.is_symlink() or ckpt_root.exists():
            ckpt_root.unlink()
        ckpt_root.symlink_to(ckpt_model)
        print(f"[EDGE setup] Created symlink: {ckpt_root} -> {ckpt_model}")
    except Exception:
        # If symlink is not allowed, copy instead
        import shutil

        shutil.copyfile(ckpt_model, ckpt_root)
        print(f"[EDGE setup] Symlink failed, copied checkpoint to {ckpt_root}")

    return ckpt_root


def setup_edge(edge_dir: Path = EDGE_DEFAULT_DIR) -> Path:
    """
    Full setup: clone EDGE repo (if needed) and ensure checkpoint is ready.
    Returns the path to the EDGE repo.
    """
    edge_dir = clone_edge_repo(edge_dir)
    ckpt = ensure_checkpoint(edge_dir)
    print(f"[EDGE setup] Ready. EDGE dir = {edge_dir}, checkpoint = {ckpt}")
    return edge_dir