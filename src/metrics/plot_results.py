from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional
import csv

import matplotlib.pyplot as plt


@dataclass
class ExperimentResult:
    name: str
    alignment: float
    jerk: float
    group: str = ""


DEFAULT_RESULTS: List[ExperimentResult] = [
    ExperimentResult("exp01", 0.155000, 0.212000, "HuBERT"),
    ExperimentResult("exp02", 0.173000, 0.296000, "HuBERT"),
    ExperimentResult("exp03", 0.160000, 0.232000, "HuBERT"),
    ExperimentResult("exp04", 0.174000, 0.340000, "HuBERT"),
    ExperimentResult("exp05", 0.169000, 0.346000, "HuBERT"),
    ExperimentResult("exp06a", 0.182000, 0.364000, "HuBERT"),
    ExperimentResult("exp06b", 0.193000, 0.389000, "HuBERT"),
    ExperimentResult("exp07", 0.186000, 0.388000, "HuBERT"),
    ExperimentResult("exp08", 0.192000, 0.348000, "HuBERT"),
    ExperimentResult("exp09", 0.200000, 0.361000, "HuBERT"),
    ExperimentResult("exp10", 0.184000, 0.237000, "WavLM"),
    ExperimentResult("exp10b", 0.164000, 0.215000, "WavLM"),
    ExperimentResult("exp11", 0.178221, 0.328077, "WavLM"),
    ExperimentResult("exp12", 0.169063, 0.268972, "WavLM"),
    ExperimentResult("exp13", 0.203643, 0.270119, "WavLM"),
]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_results_csv(csv_path: Path) -> List[ExperimentResult]:
    """
    Load results from a CSV with columns:
    experiment, alignment, jerk, group

    Required:
        experiment, alignment, jerk

    Optional:
        group
    """
    rows: List[ExperimentResult] = []

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                ExperimentResult(
                    name=row["experiment"],
                    alignment=float(row["alignment"]),
                    jerk=float(row["jerk"]),
                    group=row.get("group", ""),
                )
            )

    return rows


def save_figure(fig, output_path: Path) -> None:
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Saved: {output_path}")


def plot_alignment_bar(
    results: Iterable[ExperimentResult],
    output_path: Path,
    title: str = "Alignment Comparison Across Experiments",
) -> None:
    results = list(results)
    names = [r.name for r in results]
    values = [r.alignment for r in results]

    fig = plt.figure(figsize=(12, 5))
    plt.bar(names, values)
    plt.title(title)
    plt.xlabel("Experiment")
    plt.ylabel("Alignment")
    plt.xticks(rotation=45)
    plt.tight_layout()

    save_figure(fig, output_path)


def plot_jerk_bar(
    results: Iterable[ExperimentResult],
    output_path: Path,
    title: str = "Jerk Comparison Across Experiments",
) -> None:
    results = list(results)
    names = [r.name for r in results]
    values = [r.jerk for r in results]

    fig = plt.figure(figsize=(12, 5))
    plt.bar(names, values)
    plt.title(title)
    plt.xlabel("Experiment")
    plt.ylabel("Jerk")
    plt.xticks(rotation=45)
    plt.tight_layout()

    save_figure(fig, output_path)


def plot_tradeoff_scatter(
    results: Iterable[ExperimentResult],
    output_path: Path,
    title: str = "Alignment vs Jerk Trade-off",
    annotate: bool = True,
) -> None:
    results = list(results)
    x = [r.jerk for r in results]
    y = [r.alignment for r in results]

    fig = plt.figure(figsize=(8, 6))
    plt.scatter(x, y)

    if annotate:
        for r in results:
            plt.text(r.jerk, r.alignment, r.name)

    plt.title(title)
    plt.xlabel("Jerk")
    plt.ylabel("Alignment")
    plt.tight_layout()

    save_figure(fig, output_path)


def plot_grouped_hubert_vs_wavlm(
    results: Iterable[ExperimentResult],
    output_path: Path,
    title: str = "HuBERT vs WavLM Trade-off",
    annotate: bool = True,
) -> None:
    results = list(results)
    hubert = [r for r in results if r.group.lower() == "hubert"]
    wavlm = [r for r in results if r.group.lower() == "wavlm"]

    fig = plt.figure(figsize=(8, 6))

    if hubert:
        plt.scatter([r.jerk for r in hubert], [r.alignment for r in hubert], label="HuBERT")
    if wavlm:
        plt.scatter([r.jerk for r in wavlm], [r.alignment for r in wavlm], label="WavLM")

    if annotate:
        for r in results:
            plt.text(r.jerk, r.alignment, r.name)

    plt.title(title)
    plt.xlabel("Jerk")
    plt.ylabel("Alignment")
    plt.legend()
    plt.tight_layout()

    save_figure(fig, output_path)


def plot_lambda_sweep(
    lambdas: List[float],
    alignments: List[float],
    jerks: List[float],
    output_dir: Path,
) -> None:
    """
    Plot lambda sweep results for experiment 09.

    You should pass your final measured values here if you want exact plots.
    """
    ensure_dir(output_dir)

    fig1 = plt.figure(figsize=(7, 5))
    plt.plot(lambdas, alignments, marker="o")
    plt.title("Effect of Smoothness Weight on Alignment")
    plt.xlabel("Lambda")
    plt.ylabel("Alignment")
    plt.tight_layout()
    save_figure(fig1, output_dir / "lambda_sweep_alignment.png")

    fig2 = plt.figure(figsize=(7, 5))
    plt.plot(lambdas, jerks, marker="o")
    plt.title("Effect of Smoothness Weight on Jerk")
    plt.xlabel("Lambda")
    plt.ylabel("Jerk")
    plt.tight_layout()
    save_figure(fig2, output_dir / "lambda_sweep_jerk.png")


def make_all_default_plots(
    output_dir: Path,
    results: Optional[List[ExperimentResult]] = None,
) -> None:
    ensure_dir(output_dir)

    if results is None:
        results = DEFAULT_RESULTS

    plot_alignment_bar(
        results,
        output_dir / "alignment_bar_all_experiments.png",
    )
    plot_jerk_bar(
        results,
        output_dir / "jerk_bar_all_experiments.png",
    )
    plot_tradeoff_scatter(
        results,
        output_dir / "alignment_vs_jerk_tradeoff.png",
    )
    plot_grouped_hubert_vs_wavlm(
        results,
        output_dir / "hubert_vs_wavlm_tradeoff.png",
    )


def main() -> None:
    output_dir = Path("plots")
    make_all_default_plots(output_dir)

    # Replace these values with your final measured lambda sweep results if needed.
    lambdas = [0.0, 0.01, 0.03, 0.05, 0.1]
    alignments = [0.193000, 0.200000, 0.190000, 0.182000, 0.170000]
    jerks = [0.389000, 0.361000, 0.340000, 0.320000, 0.290000]

    plot_lambda_sweep(
        lambdas=lambdas,
        alignments=alignments,
        jerks=jerks,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
