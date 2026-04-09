#  EDGE Dance Generation (HuBERT and WavLM)

Resource-Efficient Music-to-Motion Pipeline

This repository contains a Google Colab–ready implementation of a music-to-dance generation pipeline based on the EDGE (Editable Dance Generation from Music) framework.

The main idea of the project is to replace the original Jukebox audio features used by EDGE with lighter self-supervised audio encoders such as HuBERT and WavLM. This makes the system usable in low-resource environments such as a free Google Colab T4 GPU while keeping compatibility with the original EDGE diffusion model.

The project supports:
- HuBERT-based feature extraction
- WavLM-based feature extraction
- projector adaptation
- LoRA-based encoder adaptation
- rhythm-aware and smoothness-aware experiments
- cached feature generation for efficient EDGE inference

---

#  Project Goals

The goals of this project are:

- Replace computationally expensive Jukebox embeddings with lightweight audio representations
- Enable music-to-motion generation on Google Colab
- Project 768-dimensional encoder features into the 4800-dimensional space expected by EDGE
- Support reproducible experiments with HuBERT and WavLM
- Provide a clean code structure for future fine-tuning, evaluation, and visualisation

---

#  Repository Structure

```text
.
├── README.md
├── .gitignore
├── experiments_log/
│   └── experiment metadata and saved logs
└── src/
    ├── audio_encoders/
    │   ├── hubert_cache.py
    │   ├── hubert_utils.py
    │   ├── wavlm_cache.py
    │   ├── wavlm_utils.py
    │   ├── lora.py
    │   └── cache_builder.py
    ├── colab_utils/
    │   └── install_deps.py
    ├── edge_integration/
    │   ├── edge_setup.py
    │   ├── edge_runner.py
    │   └── pytorch3d_shim/
    ├── experiments/
    │   ├── exp01_baseline_cache_only.py
    │   ├── exp02_projector_rhythm_experiment.py
    │   ├── exp03_hubert_top3_finetune.py
    │   ├── exp04_hubert_top3_e2e.py
    │   ├── exp05_hubert_top3_e2e_smooth.py
    │   ├── exp06_hubert_lora_e2e.py
    │   ├── exp07_hubert_lora_onset_smooth.py
    │   ├── exp08_hubert_lora_smoothloss.py
    │   ├── exp09_hubert_lora_lambda_sweep.py
    │   ├── train_wavlm_lora_e2e.py
    │   ├── exp10_wavlm_lora_e2e.py
    │   ├── exp10b_wavlm_baseplus_lora_e2e.py
    │   ├── exp11_wavlm_frozen.py
    │   ├── exp12_wavlm_projector_only.py
    │   └── exp13_wavlm_lora_smooth.py
    ├── metrics/
    └── models/
        └── projector.py
```

---

#  Encoder Replacement

The original EDGE pipeline uses Jukebox embeddings as conditioning features. In this project, Jukebox is replaced with lightweight audio encoders.

Two alternatives are implemented:

## **HuBERT**

HuBERT is used as a frozen or partially adapted audio encoder. Its 768-dimensional hidden states are resampled and projected into the 4800-dimensional feature space expected by EDGE.

## **WavLM**

WavLM is used as a stronger alternative encoder. It supports frozen inference, projector-only adaptation, and LoRA-based adaptation. WavLM experiments are implemented in the later experiment files, especially exp10 to exp13.

In both cases, the encoder output is mapped through a projector network before being passed to the EDGE diffusion model.

#  Google Colab Setup

The typical Colab workflow starts by mounting Google Drive and cloning the repository.

## **1. Mount Google Drive**

```python
from google.colab import drive
drive.mount("/content/drive")
```

---

## **2. Clone the Repository and Prepare Python Imports**

```python
import os
import shutil
import sys
import pathlib
import getpass

os.chdir("/content")

# Clone repository
!git clone https://github.com/kurakinanatalie/edge_dance_generating.git

# Enter repo
os.chdir("/content/edge_dance_generating")
print("CWD:", os.getcwd())

# Make src importable
REPO_DIR = pathlib.Path("/content/edge_dance_generating")
src_path = str(REPO_DIR / "src")
if src_path not in sys.path:
    sys.path.append(src_path)

print("src import path OK:", src_path in sys.path)
```

---

## **3. Install Dependencies**

```python
from colab_utils.install_deps import install_edge_dependencies
install_edge_dependencies()
```

This installs the required Python packages for audio processing and EDGE integration, including:

- transformers
- librosa
- soundfile
- p_tqdm
- tqdm
- einops
- matplotlib

---

## **4. Setup EDGE and Download Checkpoint**

```python
from edge_integration.edge_setup import setup_edge

EDGE_DIR = setup_edge()
print("EDGE_DIR:", EDGE_DIR)
```
This step:

- clones the official EDGE repository into `/content/EDGE`
- downloads the pretrained checkpoint
- prepares `checkpoint.pt`
- installs the lightweight PyTorch3D shim used by this project

---

## **5. Preparing Music Input**

Music files should be stored in a directory containing `.wav` files.

# Option A. Use music from Google Drive

```python
from pathlib import Path

MUSIC_DIR = Path("/content/drive/MyDrive/edge_music_AIST60")
print("Music folder exists:", MUSIC_DIR.exists())
print("Number of wav files:", len(list(MUSIC_DIR.glob("*.wav"))))
```

# Option B. Upload wav files directly in Colab

```python
from pathlib import Path
from colab_utils.upload_music import upload_wavs_to_dir

MUSIC_DIR = Path("/content/music")
saved = upload_wavs_to_dir(MUSIC_DIR)
print("Uploaded files:", len(saved))
```

# Option C. Create a small test folder with 2 files

```python
from pathlib import Path
import shutil
import random

SOURCE_DIR = Path("/content/drive/MyDrive/edge_music_AIST60")
TEST_MUSIC_DIR = Path("/content/music_test_small")
TEST_MUSIC_DIR.mkdir(parents=True, exist_ok=True)

for f in TEST_MUSIC_DIR.glob("*.wav"):
    f.unlink()

all_wavs = list(SOURCE_DIR.glob("*.wav"))
selected = random.sample(all_wavs, k=2)

for f in selected:
    shutil.copy(f, TEST_MUSIC_DIR / f.name)

MUSIC_DIR = TEST_MUSIC_DIR

print("Selected files:")
for f in MUSIC_DIR.glob("*.wav"):
    print("-", f.name)
```

---

## **6. Running Experiments**

This repository contains separate experiment entrypoints for HuBERT and WavLM.

The usual workflow is:

- prepare music input
- train or load projector and encoder adaptation weights
- build cached 150 x 4800 conditioning features
- run EDGE from cached features
- optionally render and preview video in Colab

---
  
# Example: WavLM LoRA and Smoothness Experiment

A typical example is experiment 13, which uses:

- WavLM
- LoRA adaptation
- smoothness regularisation
- EDGE inference from cached features
  
**1. Prepare Output Paths**

```python
from pathlib import Path

CHECKPOINT = Path(EDGE_DIR) / "checkpoint.pt"

TEST_ROOT = Path("/content/smoke_test_wavlm")
CKPT_DIR = TEST_ROOT / "checkpoints"
CACHE_DIR = TEST_ROOT / "cache_exp13"
RENDER_DIR = TEST_ROOT / "renders_exp13"
MOTION_DIR = TEST_ROOT / "motions_exp13"
META_JSON = TEST_ROOT / "meta_exp13.json"

CKPT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)
RENDER_DIR.mkdir(parents=True, exist_ok=True)
MOTION_DIR.mkdir(parents=True, exist_ok=True)

OUT_PROJECTOR = CKPT_DIR / "projector_exp13.pt"
OUT_WAVLM_LORA = CKPT_DIR / "wavlm_exp13_lora.pt"
```

**2. Train WavLM LoRA and Projector**

```python
from audio_encoders.lora import LoRAConfig
from experiments.train_wavlm_lora_e2e import train_wavlm_lora_e2e

lora_cfg = LoRAConfig(
    r=8,
    alpha=16.0,
    dropout=0.05,
    target_keywords=(
        "q_proj",
        "k_proj",
        "v_proj",
        "out_proj",
        "intermediate_dense",
        "output_dense",
    ),
)

meta = train_wavlm_lora_e2e(
    music_dir=MUSIC_DIR,
    out_projector_ckpt=OUT_PROJECTOR,
    out_wavlm_lora_ckpt=OUT_WAVLM_LORA,
    lora_cfg=lora_cfg,
    device="cuda",
    max_tracks=2,
    max_chunks_per_track=1,
    chunk_len=150,
    epochs=1,
    batch_size=1,
    lr_projector=2e-4,
    lr_lora=1e-4,
    grad_accum_steps=1,
    use_amp=True,
    lambda_var=0.1,
    smooth_lambda=0.01,
    smooth_order=2,
    seed=123,
    meta_json_path=META_JSON,
)
```

**3. Run Experiment 13 Without Rendering**

```python
from experiments.exp13_wavlm_lora_smooth import run_exp13_wavlm_lora_smooth

total_chunks = run_exp13_wavlm_lora_smooth(
    music_dir=MUSIC_DIR,
    cache_dir=CACHE_DIR,
    projector_ckpt=OUT_PROJECTOR,
    wavlm_lora_ckpt=OUT_WAVLM_LORA,
    edge_repo_dir=EDGE_DIR,
    checkpoint=CHECKPOINT,
    render_dir=RENDER_DIR,
    motion_dir=MOTION_DIR,
    device="cuda",
    chunk_len=150,
    max_tracks=2,
    max_chunks_per_track=1,
    out_length=10.0,
    cfg_target=7.5,
    no_render=True,
)

print("Done. Chunks:", total_chunks)
```

This version is useful for fast smoke tests.

**4. Run Experiment 13 With Video Rendering**

```python
from experiments.exp13_wavlm_lora_smooth import run_exp13_wavlm_lora_smooth

total_chunks = run_exp13_wavlm_lora_smooth(
    music_dir=MUSIC_DIR,
    cache_dir=CACHE_DIR,
    projector_ckpt=OUT_PROJECTOR,
    wavlm_lora_ckpt=OUT_WAVLM_LORA,
    edge_repo_dir=EDGE_DIR,
    checkpoint=CHECKPOINT,
    render_dir=RENDER_DIR,
    motion_dir=MOTION_DIR,
    device="cuda",
    chunk_len=150,
    max_tracks=2,
    max_chunks_per_track=1,
    out_length=10.0,
    cfg_target=7.5,
    no_render=False,
)

print("Done. Chunks:", total_chunks)
```

**5. Preview the Last Rendered Video**

```python
from colab_utils.preview_video import show_last_video

show_last_video(RENDER_DIR, width=720)
```

This helper:

- finds the latest rendered .mp4 or .webm
- displays it directly inside the notebook
- prints a small list of recent video files

---

# HuBERT Workflow

The repository also supports HuBERT-based experiments.

For example, a baseline HuBERT cache can be built using:

```python
from pathlib import Path
from audio_encoders.hubert_cache import build_hubert_cache

CACHE_DIR = Path("/content/cache_hubert")

total = build_hubert_cache(
    music_dir=MUSIC_DIR,
    cache_dir=CACHE_DIR,
    device="cuda"
)

print("Chunks generated:", total)
```

HuBERT experiments are implemented in:

- exp01 baseline
- exp02 rhythm-aware projector
- exp03 to exp05 fine-tuning variants
- exp06 to exp09 LoRA and regularisation variants

---

# EDGE Runner

The EDGE runner uses cached 150 x 4800 feature chunks instead of Jukebox features.

Main file:

`src/edge_integration/edge_runner.py`

The runner:

- loads cached feature chunks
- matches them with the input music files
- applies CFG scaling if requested
- calls `model.render_sample()`
- saves generated motion and optionally renders video

---

# Projector Model

Main file:

`src/models/projector.py`

Architecture:

`768 -> 1536 -> 4800`

The projector:

- maps HuBERT or WavLM features into the space expected by EDGE
- supports baseline use and fine-tuned checkpoints
- is used across both HuBERT and WavLM experiments

---

# PyTorch3D Shim

EDGE internally expects:

```python
from pytorch3d.transforms import ...
```

To avoid installing the full PyTorch3D package, this repository provides a lightweight replacement in:

`src/edge_integration/pytorch3d_shim`

It is automatically installed before EDGE is imported.

---

# Reproducibility Notes

To reproduce experiments reliably:

- always use the same directory structure for cached features
- keep checkpoint paths explicit
- use small smoke tests before full runs
- clear old cache folders before rerunning the same experiment
- use `no_render=True` for fast testing

For WavLM cache compatibility with EDGE, cached files must be stored in the format:

```text
cache_dir/
  song_name/
    0.npy
    1.npy
```

Each `.npy` file must have shape:

`(150, 4800)`

---

# Future Work

This repository is structured to support future extensions, including:

- additional WavLM training variants
- cleaner experiment automation
- quantitative evaluation utilities
- Blender-based motion visualisation
- fixing character orientation for 3D export
- presentation video generation

---

#  License

This project integrates components from Stanford's **EDGE** model.

Please refer to the original EDGE repository for licensing terms and model usage conditions.

---

#  Acknowledgements

- Stanford TML Lab — original EDGE  
- Facebook Research — HuBERT
- Microsoft Research — WavLM
