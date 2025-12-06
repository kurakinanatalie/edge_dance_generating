"""
Experiments for adapting audio representations for EDGE dance generation.

This module does NOT contain final training code yet.
Instead, it documents the planned experiments and provides
lightweight scaffolding (function stubs, comments, and TODOs).

Main idea:
    - Replace heavy Jukebox audio features with more efficient HuBERT-based features.
    - Adapt HuBERT and/or the Projector so that the resulting 4800-D embeddings
      are as useful as possible for the EDGE diffusion model.

Key research questions:
    1) How does the choice and adaptation of audio features affect motion quality?
    2) Is it enough to learn only a small Projector on top of frozen HuBERT?
    3) Do lightweight adaptation techniques (e.g., LoRA) on HuBERT bring further gains?
    4) How do different adaptation strategies compare under the same evaluation protocol?
"""

from pathlib import Path
from dataclasses import dataclass
from typing import Literal, Optional


# ---------------------------------------------------------------------------
# 1. Experiment configuration
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    """
    Configuration for a single adaptation experiment.

    This is intentionally high-level. The concrete implementation
    (optimizers, schedulers, training loops) will be added later.
    """
    name: str
    strategy: Literal[
        "baseline_frozen",
        "projector_only",
        "hubert_lora",
        "hubert_partial_unfreeze",
        "projector_plus_lora",
    ]
    description: str

    # Data
    train_music_dir: Path
    val_music_dir: Path
    cache_dir: Path

    # Model/components
    freeze_hubert: bool = True
    use_lora_on_hubert: bool = False
    train_projector: bool = False

    # Optimization (to be refined later)
    learning_rate: float = 1e-4
    batch_size: int = 4
    num_steps: int = 10_000

    # Notes / comments for the report
    notes: Optional[str] = None


# ---------------------------------------------------------------------------
# 2. Planned experiment set (high-level)
# ---------------------------------------------------------------------------

def get_planned_experiments(base_cache_dir: Path) -> list[ExperimentConfig]:
    """
    Return a list of planned experiments.

    NOTE: For now, all paths are placeholders. They should be updated once
    the final dataset organisation (train/val split) is decided.
    """
    train_music = base_cache_dir / "train_music"   # TODO: adjust
    val_music = base_cache_dir / "val_music"       # TODO: adjust
    cache_root = base_cache_dir                    # TODO: maybe split train/val caches

    return [
        ExperimentConfig(
            name="baseline_frozen",
            strategy="baseline_frozen",
            description=(
                "Baseline: use HuBERT embeddings + current Projector weights "
                "with NO additional training. This is the reference for all "
                "subsequent experiments."
            ),
            train_music_dir=train_music,
            val_music_dir=val_music,
            cache_dir=cache_root,
            freeze_hubert=True,
            use_lora_on_hubert=False,
            train_projector=False,
            notes=(
                "This corresponds to the current pipeline: HuBERT is completely frozen, "
                "and the Projector is used as initialised in models/projector.py."
            ),
        ),
        ExperimentConfig(
            name="projector_only_ft",
            strategy="projector_only",
            description=(
                "Fine-tune ONLY the Projector on top of frozen HuBERT features. "
                "HuBERT parameters remain fixed. The goal is to better map "
                "HuBERT embeddings into the 4800-D space expected by EDGE."
            ),
            train_music_dir=train_music,
            val_music_dir=val_music,
            cache_dir=cache_root,
            freeze_hubert=True,
            use_lora_on_hubert=False,
            train_projector=True,
            notes=(
                "This is the first adaptation strategy to implement, because it is "
                "computationally cheap and does not require modifying HuBERT itself."
            ),
        ),
        ExperimentConfig(
            name="hubert_lora_small",
            strategy="hubert_lora",
            description=(
                "Apply LoRA to selected layers of HuBERT while keeping the base weights frozen. "
                "The Projector is either kept fixed or lightly fine-tuned. "
                "Goal: check whether a small number of additional parameters in HuBERT "
                "improves motion quality."
            ),
            train_music_dir=train_music,
            val_music_dir=val_music,
            cache_dir=cache_root,
            freeze_hubert=False,            # base weights frozen, but LoRA params trainable
            use_lora_on_hubert=True,
            train_projector=False,          # optionally True in future variants
            notes=(
                "This requires integrating a LoRA library (e.g., PEFT) or a custom "
                "LoRA implementation for the HuBERT encoder. Exact layers and rank "
                "will be decided after analysing HuBERT's architecture."
            ),
        ),
        ExperimentConfig(
            name="hubert_partial_unfreeze",
            strategy="hubert_partial_unfreeze",
            description=(
                "Unfreeze only a small number of HuBERT layers (e.g., last N transformer blocks) "
                "and fine-tune them jointly with the Projector. "
                "Goal: compare full LoRA-style adaptation vs partial unfreezing."
            ),
            train_music_dir=train_music,
            val_music_dir=val_music,
            cache_dir=cache_root,
            freeze_hubert=False,
            use_lora_on_hubert=False,
            train_projector=True,
            notes=(
                "This experiment requires careful regularisation and probably a small learning "
                "rate for HuBERT layers to avoid catastrophic forgetting."
            ),
        ),
        ExperimentConfig(
            name="projector_plus_lora_combo",
            strategy="projector_plus_lora",
            description=(
                "Combined strategy: fine-tune the Projector and apply LoRA to HuBERT. "
                "Goal: check whether learning both the mapping layer and small HuBERT "
                "adapters yields better motion quality than either alone."
            ),
            train_music_dir=train_music,
            val_music_dir=val_music,
            cache_dir=cache_root,
            freeze_hubert=False,
            use_lora_on_hubert=True,
            train_projector=True,
            notes=(
                "This is a more advanced experiment and should be attempted only after "
                "the simpler baselines (projector_only_ft and hubert_lora_small) are understood."
            ),
        ),
    ]


# ---------------------------------------------------------------------------
# 3. Placeholder for training / evaluation infrastructure
# ---------------------------------------------------------------------------

def describe_experiments(base_cache_dir: Path) -> None:
    """
    Utility function that prints a human-readable summary of planned experiments.

    This can be called from a notebook to show the current experimental design
    in a compact way (useful for supervision meetings and the final report).
    """
    exps = get_planned_experiments(base_cache_dir)
    print("Planned experiments:")
    for exp in exps:
        print("=" * 80)
        print(f"Name:        {exp.name}")
        print(f"Strategy:    {exp.strategy}")
        print(f"Freeze HuBERT:       {exp.freeze_hubert}")
        print(f"Use LoRA on HuBERT:  {exp.use_lora_on_hubert}")
        print(f"Train Projector:     {exp.train_projector}")
        print(f"LR / batch / steps:  {exp.learning_rate} / {exp.batch_size} / {exp.num_steps}")
        print(f"Train music dir:     {exp.train_music_dir}")
        print(f"Val music dir:       {exp.val_music_dir}")
        print(f"Cache root:          {exp.cache_dir}")
        print("Description:")
        print("  ", exp.description)
        if exp.notes:
            print("Notes:")
            print("  ", exp.notes)


# ---------------------------------------------------------------------------
# 4. TODO: future work (to be filled during the project)
# ---------------------------------------------------------------------------

# TODO:
# - Decide on the exact dataset organisation (train/validation split) and
#   update train_music_dir / val_music_dir / cache_dir accordingly.
#
# - Implement a small training loop for the 'projector_only_ft' experiment:
#     * load cached HuBERT features
#     * define a suitable objective (e.g., reconstruction of Jukebox-like features,
#       alignment with internal EDGE embeddings, or another proxy target)
#     * optimise only the Projector parameters
#
# - Implement LoRA-based adaptation for HuBERT (starting with hubert_lora_small):
#     * choose which transformer layers to adapt
#     * decide LoRA rank and placement (attention / FFN)
#     * integrate with the existing feature extraction pipeline
#
# - Define an evaluation protocol for motion quality:
#     * objective metrics (if available in EDGE or related work)
#     * subjective / qualitative analysis (visual inspection, side-by-side videos)
#
# - Record experimental results (tables/plots) for the report:
#     * baseline vs projector_only_ft vs hubert_lora_small vs others
#     * discussion of trade-offs (quality vs compute, stability, etc.)