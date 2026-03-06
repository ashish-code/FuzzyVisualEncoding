"""
extract_pca_features.py — PCA-reduce pooled SIFT features to a lower dimension.

Reads a pooled (all-category) tab-delimited feature matrix, fits a PCA model
to reduce it to ``--lower-dim`` dimensions (default 10), writes the reduced
matrix, and then transforms and writes each per-category feature file.

This step corresponds to the dimensionality reduction prior to fuzzy clustering
described in Section III-B of the ICIP 2012 paper.  PCA removes correlated
dimensions from 128-d SIFT space, keeping most variance while making the
subsequent clustering (GK/GG) computationally tractable.

Original MATLAB equivalent: none (pure pre-processing utility)
Original Python version: writePCAfeature.py (2011, Python 2)

Migration notes
---------------
- Replaced ``optparse.OptionParser`` with ``argparse``.
- Removed ``print(print ...)`` double-print syntax error.
- Replaced deprecated ``np.int`` dtype alias with builtin ``int``.
- Used ``pathlib.Path`` throughout.
- Added persistent PCA model serialisation (``joblib``) so transforms can be
  re-applied to test sets without re-fitting.

Usage
-----
    python scripts/extract_pca_features.py \\
        --root-dir /path/to/data \\
        --dataset VOC2006 \\
        --lower-dim 10
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)

_TAB_EXT = ".tab"


def _load_category_map(dataset_root: Path) -> dict[str, int]:
    """Parse catidlist.txt → {category_name: category_id}."""
    cat_file = dataset_root / "catidlist.txt"
    if not cat_file.is_file():
        raise FileNotFoundError(f"Category list not found: {cat_file}")
    cat_names = np.genfromtxt(cat_file, delimiter=",", dtype=str, usecols=[0])
    cat_ids = np.genfromtxt(cat_file, delimiter=",", dtype=int, usecols=[1])
    return dict(zip(cat_names.tolist(), cat_ids.tolist()))


def extract_pca_features(
    root_dir: Path,
    dataset: str,
    lower_dim: int = 10,
    max_samples_per_cat: int = 100,
) -> None:
    """Fit PCA on pooled features and write reduced per-category files.

    Parameters
    ----------
    root_dir : Path
        Parent data directory.
    dataset : str
        Dataset subdirectory name.
    lower_dim : int, default 10
        Target PCA dimensionality.
    max_samples_per_cat : int, default 100
        Maximum number of feature vectors written per category after PCA.
        Keeps output files manageable; set to -1 for no limit.
    """
    dataset_root = root_dir / dataset
    feature_dir = dataset_root / "Fuzzy" / "FeatureUni"
    tab_dir = dataset_root / "Fuzzy" / "Feature"

    # ------------------------------------------------------------------ Fit PCA
    pooled_path = feature_dir / f"{dataset}{_TAB_EXT}"
    if not pooled_path.is_file():
        raise FileNotFoundError(
            f"Pooled feature file not found: {pooled_path}\n"
            "Run extract_feature_vectors.py first."
        )

    logger.info("Loading pooled features from %s ...", pooled_path)
    fvector = np.loadtxt(pooled_path, dtype=int, delimiter="\t")
    logger.info("  Shape: %s", fvector.shape)

    pca = PCA(n_components=lower_dim)
    pca.fit(fvector)
    explained = pca.explained_variance_ratio_.sum()
    logger.info(
        "PCA fitted: %d → %d dims  (%.1f%% variance retained)",
        fvector.shape[1],
        lower_dim,
        explained * 100,
    )

    # Write reduced pooled matrix
    fv_reduced = pca.transform(fvector)
    out_pooled = feature_dir / f"{dataset}{lower_dim}{_TAB_EXT}"
    np.savetxt(out_pooled, fv_reduced, fmt="%f", delimiter="\t")
    logger.info("Saved pooled reduced features → %s", out_pooled)

    # Optionally save the PCA model for later reuse on test data
    try:
        import joblib
        model_path = feature_dir / f"pca_{lower_dim}d.joblib"
        joblib.dump(pca, model_path)
        logger.info("Saved PCA model → %s", model_path)
    except ImportError:
        logger.warning("joblib not available; PCA model not persisted.")

    # ------------------------------------------------- Transform per-category
    cat_map = _load_category_map(dataset_root)
    for cat_name in sorted(cat_map):
        cat_src = tab_dir / f"{cat_name}{_TAB_EXT}"
        if not cat_src.is_file():
            logger.warning("Category file missing, skipping: %s", cat_src)
            continue

        logger.info("Transforming %s ...", cat_name)
        cat_data = np.loadtxt(cat_src, dtype=int, delimiter="\t")
        cat_reduced = pca.transform(cat_data)

        if max_samples_per_cat > 0:
            cat_reduced = cat_reduced[:max_samples_per_cat]

        cat_dst = tab_dir / f"{cat_name}{lower_dim}{_TAB_EXT}"
        np.savetxt(cat_dst, cat_reduced, fmt="%f", delimiter="\t")
        logger.info("  → %s  (%d vectors)", cat_dst, len(cat_reduced))


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="PCA-reduce pooled and per-category SIFT features.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--root-dir", required=True, type=Path)
    p.add_argument("--dataset", default="VOC2006")
    p.add_argument("--lower-dim", type=int, default=10, help="Target PCA dimensionality")
    p.add_argument("--max-samples", type=int, default=100, help="Max vectors per category (-1 for all)")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    extract_pca_features(args.root_dir, args.dataset, args.lower_dim, args.max_samples)
    return 0


if __name__ == "__main__":
    sys.exit(main())
