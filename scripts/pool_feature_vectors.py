"""
pool_feature_vectors.py — Pool per-category descriptors into a single matrix.

Samples up to ``--n-cluster-sample`` SIFT vectors uniformly from all categories
and writes them to a single pooled .tab file.  This pooled matrix is used to
fit the PCA transform (extract_pca_features.py) and the fuzzy dictionary
(FuzzyDictionary.fit).

Original Python version: writeFeatureVector.py (2011, Python 2)

Migration notes
---------------
- Replaced ``optparse`` with ``argparse``.
- Replaced ``clusterData == None`` with ``clusterData is None``.
- Replaced deprecated ``np.int16`` array construction with explicit dtype kwarg.
- Used ``pathlib.Path`` throughout.
- Added structured logging.

Usage
-----
    python scripts/pool_feature_vectors.py \\
        --root-dir /path/to/data \\
        --dataset VOC2006 \\
        --n-cluster-sample 50000 \\
        --descriptor sift
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

DESCRIPTOR_DIM: dict[str, int] = {
    "sift": 128,
    "surf": 64,
}
_DESCRIPTOR_OFFSET = 2  # first two columns are frame_id, image_id


def _load_category_map(dataset_root: Path) -> dict[str, int]:
    """Parse catidlist.txt → {category_name: category_id}."""
    cat_file = dataset_root / "catidlist.txt"
    if not cat_file.is_file():
        raise FileNotFoundError(f"Category list not found: {cat_file}")
    cat_names = np.genfromtxt(cat_file, delimiter=",", dtype=str, usecols=[0])
    cat_ids = np.genfromtxt(cat_file, delimiter=",", dtype=int, usecols=[1])
    return dict(zip(cat_names.tolist(), cat_ids.tolist()))


def pool_feature_vectors(
    root_dir: Path,
    dataset: str,
    n_cluster_sample: int = 50_000,
    descriptor: str = "sift",
    random_state: int = 42,
) -> None:
    """Pool and write a representative sample of descriptor vectors.

    Samples ``n_cluster_sample // n_categories`` vectors from each category
    (or all vectors if a category has fewer) and concatenates them into a
    single matrix saved as ``<dataset>.tab`` in the FeatureUni directory.

    Parameters
    ----------
    root_dir : Path
        Parent data directory.
    dataset : str
        Dataset name.
    n_cluster_sample : int, default 50_000
        Total number of descriptors to pool across all categories.
    descriptor : str, default 'sift'
        Feature descriptor type.
    random_state : int, default 42
        Seed for reproducible sub-sampling.
    """
    rng = np.random.default_rng(random_state)
    dim = DESCRIPTOR_DIM.get(descriptor)
    if dim is None:
        raise ValueError(f"Unknown descriptor '{descriptor}'.")

    dataset_root = root_dir / dataset
    feature_dir = dataset_root / "FeatureMatrix"
    output_dir = dataset_root / "Fuzzy" / "FeatureUni"
    output_dir.mkdir(parents=True, exist_ok=True)

    cat_map = _load_category_map(dataset_root)
    n_categories = len(cat_map)
    n_per_cat = max(1, n_cluster_sample // n_categories)
    logger.info(
        "Pooling ~%d samples/category from %d categories ...",
        n_per_cat,
        n_categories,
    )

    col_range = np.arange(_DESCRIPTOR_OFFSET, _DESCRIPTOR_OFFSET + dim)
    chunks: list[np.ndarray] = []

    for cat_name in sorted(cat_map):
        src = feature_dir / f"{cat_name}.{descriptor}"
        if not src.is_file():
            logger.warning("Missing descriptor file, skipping: %s", src)
            continue

        logger.info("Reading %s ...", cat_name)
        cat_data = np.loadtxt(src, dtype=np.int16, usecols=col_range)

        if len(cat_data) <= n_per_cat:
            sample = cat_data
        else:
            idx = rng.choice(len(cat_data), n_per_cat, replace=False)
            sample = cat_data[idx]

        chunks.append(sample)
        logger.info("  %d vectors sampled (total so far: %d)", len(sample), sum(len(c) for c in chunks))

    if not chunks:
        logger.error("No descriptor files found.  Check --root-dir and --dataset.")
        return

    pooled = np.concatenate(chunks, axis=0)
    out_path = output_dir / f"{dataset}.tab"
    np.savetxt(out_path, pooled, fmt="%d", delimiter="\t")
    logger.info("Saved pooled matrix (%d × %d) → %s", pooled.shape[0], pooled.shape[1], out_path)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Pool per-category SIFT vectors into a single clustering matrix.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--root-dir", required=True, type=Path)
    p.add_argument("--dataset", default="VOC2006")
    p.add_argument("--n-cluster-sample", type=int, default=50_000)
    p.add_argument("--descriptor", default="sift", choices=list(DESCRIPTOR_DIM))
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    pool_feature_vectors(
        args.root_dir,
        args.dataset,
        args.n_cluster_sample,
        args.descriptor,
        args.random_state,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
