#  EDGE Dance Generation – HuBERT-based Music-to-Motion Pipeline

This repository contains a clean, modular re-implementation of the original  
**EDGE (Efficient Diffusion for Dance Generation)** preprocessing pipeline.

The goal of this project is to:

- replace heavy Jukebox audio features with **HuBERT-based features**,
- project HuBERT embeddings into the dimensionality expected by EDGE,
- generate dance motion using cached features,
- make the entire workflow reproducible in **Google Colab**,
- prepare the system for future fine-tuning (freezing, LoRA, adapters, etc.).

---

#  Project Structure

```
src/
  audio_encoders/
  models/
  edge_integration/
  colab_utils/
notebooks/
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

## **6. View the Last Rendered Video**

```python
from IPython.display import Video
from pathlib import Path

videos = sorted((EDGE_DIR / "renders").rglob("*.mp4"))
Video(str(videos[-1]), width=720)
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

A minimal wrapper that:

- loads cached (150×4800) features  
- applies CFG scale if available  
- calls `model.render_sample()`  
- outputs motion sequences and rendered videos  

---

#  Future Work

This structure supports future extensions:

- HuBERT fine-tuning  
- Layer freezing experiments  
- LoRA adapters  
- Alternative audio encoders  
- Quantitative evaluation  
- Ablations  

These will later be placed in:

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
