# EDGE Dance Generation – HuBERT-based Music-to-Motion Pipeline

This repository contains a clean, modular re-implementation of the original  
**EDGE (Efficient Diffusion for Dance Generation)** preprocessing pipeline.

The goal of this project is to:

- replace heavy Jukebox audio features with **HuBERT-based features**,
- project HuBERT embeddings into the dimensionality expected by EDGE,
- generate dance motion using cached features,
- make the entire workflow reproducible in **Google Colab**,
- prepare the system for future fine-tuning (freezing, LoRA, adapters, etc.).

All heavy operations (audio encoding, checkpoint download, rendering)  
are performed inside Colab — the repository itself contains **model-agnostic code only**.

---

# Project Structure

src/
audio_encoders/
hubert_utils.py # time_resample(), to_chunks()
hubert_cache.py # HuBERT feature extraction + 4800-D projection
models/
projector.py # Lightweight adapter 768 → 4800
edge_integration/
pytorch3d_shim/ # Minimal replacement for pytorch3d.transforms
init.py
transforms.py
edge_runner.py # Run EDGE using cached (150×4800) features
edge_setup.py # Clone EDGE repo + download checkpoint (with fallback)
colab_utils/
install_deps.py # Install required Python dependencies

This structure separates:

- **audio processing**,  
- **models**,  
- **EDGE integration**,  
- **Colab-specific utilities**.

---

# Getting Started (Google Colab)

This section describes the **exact order of steps** required to run the project in Google Colab.

---

## 1 Install Dependencies

```python
from colab_utils.install_deps import install_edge_dependencies
install_edge_dependencies()

This installs:

p_tqdm

tqdm

soundfile

librosa

einops

matplotlib

transformers

These are required for audio encoding, HuBERT inference, and EDGE rendering.

## 2 Clone and Prepare the Official EDGE Model

from edge_integration.edge_setup import setup_edge
EDGE_DIR = setup_edge()


setup_edge():

clones the official Stanford EDGE repository into /content/EDGE,

downloads the model checkpoint using download_model.sh,

if the checkpoint is invalid or too small, downloads a correct version via gdown,

creates/updates checkpoint.pt in the EDGE root directory.

## 3 Upload Music Files (.wav only)
from google.colab import files
from pathlib import Path
import shutil

MUSIC_DIR = Path("/content/music")
MUSIC_DIR.mkdir(exist_ok=True)

uploaded = files.upload()
for name in uploaded:
    if name.lower().endswith(".wav"):
        shutil.move(name, MUSIC_DIR / name)

## 4 Build HuBERT Feature Cache (150 × 4800)
from pathlib import Path
from audio_encoders.hubert_cache import build_hubert_cache

CACHE_DIR = Path("/content/cache")

total = build_hubert_cache(
    music_dir=MUSIC_DIR,
    cache_dir=CACHE_DIR,
    device="cuda"  # CPU also works, but slower
)

print("Chunks generated:", total)


Each .wav file produces multiple 150-frame feature slices
expected by the EDGE diffusion model.

## 5 Generate Dance Motion from Cached Features
from edge_integration.edge_runner import run_edge_from_cache
from pathlib import Path

RENDER_DIR = EDGE_DIR / "renders"
CHECKPOINT = EDGE_DIR / "checkpoint.pt"

run_edge_from_cache(
    edge_repo_dir=EDGE_DIR,
    feature_cache_dir=CACHE_DIR,
    music_dir=MUSIC_DIR,
    checkpoint=CHECKPOINT,
    render_dir=RENDER_DIR,
    out_length=10.0,
    save_motions=True,
    motion_save_dir=EDGE_DIR / "eval/motions",
    no_render=False,
    cfg_target=7.5
)


Output motion data and render videos are saved to:

/content/EDGE/renders/

## 6 View the Last Rendered Video
from IPython.display import Video
from pathlib import Path

videos = sorted((EDGE_DIR / "renders").rglob("*.mp4"))
Video(str(videos[-1]), width=720)

## Technical Overview

HuBERT feature extraction

build_hubert_cache() uses:

Facebook hubert-base-ls960,

16 kHz audio,

~50 Hz representations,

linear resampling to 30 Hz,

feature normalization,

projection to 4800 dimensions (EDGE-compatible).

## Projector model

Located in models/projector.py
Architecture: 768 → 1536 → 4800 with GELU and orthogonal initialization.

## pytorch3d shim

EDGE internally depends on pytorch3d.transforms.
To avoid installing full PyTorch3D in Colab, the project provides a lightweight
drop-in replacement inside edge_integration/pytorch3d_shim.
The runner automatically registers this shim before importing EDGE.

## EDGE runner

A minimal inference wrapper that:

loads cached features,

injects CFG/guidance scale if present,

calls model.render_sample(),

writes video output.

## Future Work

This codebase is structured to support:

HuBERT fine-tuning

Layer freezing strategies

LoRA lightweight adapters

Alternative audio encoders

Quantitative evaluation of generated motion

Ablation studies

These will be implemented in future modules under:

src/experiments/

## License

This project integrates components from the Stanford EDGE repository.
Please refer to the original EDGE license for model usage conditions.

## Acknowledgements

Stanford TML Lab — original EDGE model

Facebook Research — HuBERT
