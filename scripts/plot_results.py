"""
plot_results.py — Visualise fuzzy encoding classification results.

Consolidates three separate 2011-era scripts:

    plotfuzzyresultCategory.py  → :func:`plot_by_category`
    plotfuzzyresultDataset.py   → :func:`plot_by_dataset`
    plotfuzzyresultdictsize.py  → :func:`plot_by_dict_size`

All three functions read pre-computed ``.acc`` files (one accuracy value per
category or per fold, space-delimited) from the result directory and produce
bar-chart comparisons between BoF (k-means), FCM, and GK encoding methods.

Original Python versions: plotfuzzy*.py (2011, Python 2)

Migration notes
---------------
- Merged three scripts into a single CLI with sub-commands.
- Replaced ``OptionParser`` with ``argparse`` sub-commands.
- Replaced ``print accResult`` (Python 2 print statement) with ``print()``.
- Replaced deprecated ``np.float`` / ``np.int`` dtype aliases with builtins.
- Replaced ``np.zeros(..., np.float)`` with ``np.zeros(..., float)``.
- Added ``tight_layout()`` for better figure formatting.
- Replaced hardcoded Surrey filesystem paths with ``--root-dir`` / ``--output-dir`` args.

Usage
-----
    # Accuracy by category (bar chart per category for one dataset/dict-size)
    python scripts/plot_results.py category \\
        --root-dir /path/to/data --dataset VOC2006 --word 16

    # Accuracy by dataset (bar chart per dataset for one dict-size)
    python scripts/plot_results.py dataset \\
        --root-dir /path/to/data --word 16

    # Accuracy vs dictionary size (for Caltech101)
    python scripts/plot_results.py dictsize \\
        --root-dir /path/to/data --dataset Caltech101
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

METHODS = ["Kmeans", "FCM", "GK"]
WORDS = [16, 32, 64, 128, 256, 512]
DATASETS = ["Caltech101", "VOC2006", "VOC2010", "Caltech256", "Scene15"]
METHOD_COLORS = {"Kmeans": "#f5c542", "FCM": "#4f9e55", "GK": "#2d3a8c"}

BAR_WIDTH = 0.25


def _load_acc(result_dir: Path, method: str, dim: int, word: int) -> np.ndarray | None:
    """Load accuracy values from a .acc file; return None if missing."""
    path = result_dir / f"{method}{dim}{word}.acc"
    if not path.is_file():
        logger.debug("Missing: %s", path)
        return None
    try:
        return np.loadtxt(path, dtype=float, delimiter=" ")
    except Exception as exc:
        logger.warning("Could not read %s: %s", path, exc)
        return None


def _load_category_names(dataset_root: Path) -> list[str]:
    """Parse catidlist.txt and return ordered category names."""
    cat_file = dataset_root / "catidlist.txt"
    if not cat_file.is_file():
        return []
    return np.genfromtxt(cat_file, delimiter=",", dtype=str, usecols=[0]).tolist()


def plot_by_category(
    root_dir: Path,
    dataset: str,
    word: int,
    dim: int = 2,
    output_dir: Path | None = None,
) -> None:
    """Bar chart of per-category accuracy for BoF vs GK at a fixed dict size.

    Parameters
    ----------
    root_dir : Path
        Parent data directory.
    dataset : str
        Dataset name (must have catidlist.txt and Fuzzy/Result/*.acc).
    word : int
        Dictionary size to plot.
    dim : int, default 2
        Feature dimension tag embedded in .acc filenames.
    output_dir : Path or None
        Where to save the PNG.  Defaults to <root_dir>/<dataset>/Fuzzy/Result/.
    """
    dataset_root = root_dir / dataset
    result_dir = dataset_root / "Fuzzy" / "Result"
    cat_names = _load_category_names(dataset_root)
    n_cat = len(cat_names)
    if n_cat == 0:
        logger.error("No categories found for %s", dataset)
        return

    acc_result = np.zeros((len(METHODS), n_cat), dtype=float)
    for i, method in enumerate(METHODS):
        acc = _load_acc(result_dir, method, dim, word)
        if acc is not None:
            acc_result[i] = acc

    logger.info("Per-category accuracy:\n%s", acc_result)

    ind = np.arange(n_cat)
    fig, ax = plt.subplots(figsize=(max(10, n_cat * 0.6), 5))
    ax.bar(ind, acc_result[0], BAR_WIDTH, color=METHOD_COLORS["Kmeans"], label="BoF")
    ax.bar(ind + BAR_WIDTH, acc_result[2], BAR_WIDTH, color=METHOD_COLORS["GK"], label="GK")
    ax.set_xticks(ind + BAR_WIDTH / 2)
    ax.set_xticklabels(cat_names, rotation=35, ha="right", fontsize="small")
    ax.set_xlabel("Visual category", fontsize="large")
    ax.set_ylabel("mAcc", fontsize="large")
    ax.set_title(f"{dataset}  (dict={word})", fontsize="large")
    ax.legend()
    ax.set_ylim([max(0, acc_result.min() - 0.05), min(1, acc_result.max() + 0.05)])
    fig.tight_layout()

    out_dir = output_dir or result_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{dataset}_{word}_by_category.png"
    fig.savefig(out_path, dpi=150)
    logger.info("Saved → %s", out_path)
    plt.show()


def plot_by_dataset(
    root_dir: Path,
    word: int,
    datasets: list[str] | None = None,
    dim: int = 2,
    output_dir: Path | None = None,
) -> None:
    """Bar chart comparing BoF / FCM / GK mean accuracy across datasets.

    Parameters
    ----------
    root_dir : Path
    word : int
        Dictionary size.
    datasets : list[str] or None
        Dataset names to include.  Defaults to DATASETS constant.
    dim : int, default 2
    output_dir : Path or None
    """
    if datasets is None:
        datasets = DATASETS

    acc_result = np.zeros((len(METHODS), len(datasets)), dtype=float)
    for i, method in enumerate(METHODS):
        for j, dataset in enumerate(datasets):
            result_dir = root_dir / dataset / "Fuzzy" / "Result"
            acc = _load_acc(result_dir, method, dim, word)
            if acc is not None:
                acc_result[i, j] = float(np.mean(acc))

    logger.info("Cross-dataset accuracy:\n%s", acc_result)

    ind = np.arange(len(datasets))
    fig, ax = plt.subplots(figsize=(10, 5))
    for k, (method, color) in enumerate(METHOD_COLORS.items()):
        ax.bar(ind + k * BAR_WIDTH, acc_result[k], BAR_WIDTH, color=color, label=method)
    ax.set_xticks(ind + BAR_WIDTH)
    ax.set_xticklabels(datasets, rotation=0, fontsize="medium")
    ax.set_xlabel("Visual dataset", fontsize="large")
    ax.set_ylabel("Mean accuracy", fontsize="large")
    ax.set_title(f"Cross-dataset comparison  (dict={word})", fontsize="large")
    ax.legend()
    ax.set_ylim([max(0, acc_result.min() - 0.05), min(1, acc_result.max() + 0.05)])
    fig.tight_layout()

    out_dir = output_dir or root_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"perf_datasets_{word}.png"
    fig.savefig(out_path, dpi=150)
    logger.info("Saved → %s", out_path)
    plt.show()


def plot_by_dict_size(
    root_dir: Path,
    dataset: str,
    words: list[int] | None = None,
    dim: int = 2,
    output_dir: Path | None = None,
) -> None:
    """Line/bar chart of accuracy vs dictionary size (number of visual words).

    Parameters
    ----------
    root_dir : Path
    dataset : str
    words : list[int] or None
        Dictionary sizes to include.  Defaults to WORDS constant.
    dim : int, default 2
    output_dir : Path or None
    """
    if words is None:
        words = WORDS

    dataset_root = root_dir / dataset
    result_dir = dataset_root / "Fuzzy" / "Result"

    acc_result = np.zeros((len(METHODS), len(words)), dtype=float)
    for i, method in enumerate(METHODS):
        for j, word in enumerate(words):
            acc = _load_acc(result_dir, method, dim, word)
            if acc is not None:
                acc_result[i, j] = float(np.mean(acc))

    logger.info("Accuracy vs dict size:\n%s", acc_result)

    ind = np.arange(len(words))
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(ind, acc_result[0], BAR_WIDTH, color=METHOD_COLORS["Kmeans"], label="BoF")
    ax.bar(ind + BAR_WIDTH, acc_result[2], BAR_WIDTH, color=METHOD_COLORS["GK"], label="GK")
    ax.set_xticks(ind + BAR_WIDTH / 2)
    ax.set_xticklabels([str(w) for w in words])
    ax.set_xlabel("Dictionary size (# visual words)", fontsize="large")
    ax.set_ylabel("Mean accuracy", fontsize="large")
    ax.set_title(f"{dataset} — accuracy vs dictionary size", fontsize="large")
    ax.legend()
    ax.set_ylim([max(0, acc_result.min() - 0.05), min(1, acc_result.max() + 0.05)])
    fig.tight_layout()

    out_dir = output_dir or result_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{dataset}_dictsize.png"
    fig.savefig(out_path, dpi=150)
    logger.info("Saved → %s", out_path)
    plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Plot fuzzy encoding classification results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--root-dir", required=True, type=Path, help="Data root directory")
    p.add_argument("--output-dir", type=Path, default=None, help="Where to save PNG files")
    p.add_argument("-v", "--verbose", action="store_true")

    sub = p.add_subparsers(dest="command", required=True)

    # sub-command: category
    sc = sub.add_parser("category", help="Per-category accuracy for one dataset/dict-size")
    sc.add_argument("--dataset", default="VOC2006")
    sc.add_argument("--word", type=int, default=16, help="Dictionary size")

    # sub-command: dataset
    sd = sub.add_parser("dataset", help="Cross-dataset accuracy comparison")
    sd.add_argument("--word", type=int, default=16, help="Dictionary size")
    sd.add_argument("--datasets", nargs="+", default=None)

    # sub-command: dictsize
    sz = sub.add_parser("dictsize", help="Accuracy vs dictionary size")
    sz.add_argument("--dataset", default="Caltech101")
    sz.add_argument("--words", nargs="+", type=int, default=None)

    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    if args.command == "category":
        plot_by_category(args.root_dir, args.dataset, args.word, output_dir=args.output_dir)
    elif args.command == "dataset":
        plot_by_dataset(args.root_dir, args.word, datasets=args.datasets, output_dir=args.output_dir)
    elif args.command == "dictsize":
        plot_by_dict_size(args.root_dir, args.dataset, words=args.words, output_dir=args.output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
