# EDGE Dance Generation (HuBERT & WavLM)

Resource-Efficient Music-to-Motion Pipeline

This repository provides a clean, modular and Google Colab–ready implementation of a music-to-dance generation pipeline based on the EDGE (Editable Dance Generation from Music) framework.

The project focuses on low-resource generative modelling, replacing heavy Jukebox audio features with lightweight self-supervised encoders such as HuBERT and WavLM, while preserving compatibility with the original EDGE diffusion model.

The goal of this project is to:

- Replace computationally expensive Jukebox embeddings with lightweight audio representations  
- Enable dance generation on a free Google Colab T4 GPU  
- Introduce a projection module (768 → 4800) for EDGE compatibility  
- Support feature caching for fast experimentation  
- Provide a clean structure for future research (LoRA, fine-tuning, evaluation)  

---

# Project Structure

```
src/
audio_encoders/ # HuBERT / WavLM feature extraction
models/ # Projection networks
edge_integration/ # EDGE setup and execution
colab_utils/ # Colab helpers

notebooks/ # Experiment notebooks
```

---

#  Getting Started (Google Colab)

## **1. Install Dependencies**

```python
from colab_utils.install_deps import install_edge_dependencies
install_edge_dependencies()
```

This installs:

- p_tqdm  
- tqdm  
- soundfile  
- librosa  
- einops  
- matplotlib  
- transformers  

---

## **2. Clone and Prepare the Official EDGE Model**

```python
from edge_integration.edge_setup import setup_edge
EDGE_DIR = setup_edge()
```

`setup_edge()`:

- clones the official Stanford EDGE repo into `/content/EDGE`  
- downloads the model checkpoint  
- falls back to gdown if needed  
- prepares `checkpoint.pt`

---

## **3. Upload Music Files (.wav only)**

```python
from google.colab import files
from pathlib import Path
import shutil

MUSIC_DIR = Path("/content/music")
MUSIC_DIR.mkdir(exist_ok=True)

uploaded = files.upload()
for name in uploaded:
    if name.lower().endswith(".wav"):
        shutil.move(name, MUSIC_DIR / name)
```

---

## **4. Build HuBERT Feature Cache (150 × 4800)**

```python
from pathlib import Path
from audio_encoders.hubert_cache import build_hubert_cache

CACHE_DIR = Path("/content/cache")

total = build_hubert_cache(
    music_dir=MUSIC_DIR,
    cache_dir=CACHE_DIR,
    device="cuda"
)

print("Chunks generated:", total)
```

Each `.wav` file produces several 150-frame slices used as diffusion model conditions.

---

## **5. Generate Dance Motion from Cached Features**

```python
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
```

Output files are saved to:

```
/content/EDGE/renders/
```

---

#  Technical Overview

## **HuBERT Feature Extraction**

`build_hubert_cache()` uses:

- facebook/hubert-base-ls960  
- 16 kHz audio  
- ~50 Hz latent representations  
- resampling to 30 Hz  
- feature normalization  
- projection to 4800-D (EDGE-compatible)

---

## **Projector Model**

Located in `models/projector.py`

```
768 → 1536 → 4800
GELU activations
Orthogonal initialization
```

---

## **pytorch3d shim**

EDGE internally requires:

```
from pytorch3d.transforms import ...
```

To avoid installing PyTorch3D (very heavy),  
this repo includes a **lightweight drop-in replacement**:

```
edge_integration/pytorch3d_shim
```

Automatically activated by the runner.

---

## **EDGE Runner**

The runner:

- loads cached (150×4800) features
- applies CFG scale if available
- calls model.render_sample()
- outputs motion sequences and rendered videos

---

#  Future Work

This structure supports future extensions:

- HuBERT fine-tuning
- LoRA adapters
- Alternative audio encoders
- Quantitative evaluation
- Ablation studies  

Planned location:
```
src/experiments/
```

---

#  License

This project integrates components from Stanford's **EDGE** model.  
Refer to the original EDGE license for model usage conditions.

---

#  Acknowledgements

- Stanford TML Lab — original EDGE  
- Facebook Research — HuBERT
- Microsoft Research — WavLM
