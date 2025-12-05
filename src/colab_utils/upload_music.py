from google.colab import files
from pathlib import Path
import shutil
import os


def upload_wavs_to_dir(target_dir: Path):
    """
    Open a file chooser in Colab and upload .wav files into target_dir.
    Non-.wav files are skipped.
    Returns a list of saved file paths.
    """
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    print("Upload one or more .wav files (10–30 seconds each). MP3 is not accepted.")
    uploaded = files.upload()

    saved_paths = []
    for name in uploaded:
        if not name.lower().endswith(".wav"):
            print(f"Skipped (not .wav): {name}")
            try:
                os.remove(name)
            except Exception:
                pass
            continue
        dst = target_dir / name
        shutil.move(name, dst)
        saved_paths.append(dst)

    print("Done. Files in", target_dir)
    for p in sorted(target_dir.glob("*.wav")):
        print(" -", p.name)

    return saved_paths