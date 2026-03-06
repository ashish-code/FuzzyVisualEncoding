"""
extract_feature_vectors.py — Collect per-category SIFT descriptors into .tab files.

Reads raw SIFT descriptor files (one per category, binary whitespace-delimited
with columns: frame_id, image_id, sift_1..sift_128) and writes per-category
tab-delimited feature matrix files used as input to the fuzzy dictionary
pipeline.

Original MATLAB equivalent: calcFuzzyCoeff.m (data-prep portion)
Original Python version: writeCategoryVector.py (2011, Python 2)

Migration notes
---------------
- Replaced ``optparse.OptionParser`` with ``argparse``.
- Replaced deprecated ``np.int`` dtype alias with builtin ``int``.
- Used ``pathlib.Path`` for all I/O path manipulation.
- Added structured logging and progress reporting.

Usage
-----
    python scripts/extract_feature_vectors.py \\
        --root-dir /path/to/data \\
        --dataset VOC2006 \\
        --descriptor sift

Directory layout expected under <root-dir>/<dataset>/:
    FeatureMatrix/<category>.<desc>   — raw descriptor file
    catidlist.txt                     — comma-separated (category_name, cat_id)

Output written to:
    <root-dir>/<dataset>/Fuzzy/Feature/<category>.tab
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Column offset of the first descriptor element in the raw .sift file.
# Columns 0 and 1 are frame_id and image_id respectively.
_DESCRIPTOR_OFFSET = 2

DESCRIPTOR_DIM: dict[str, int] = {
    "sift": 128,
    "surf": 64,
}


def _load_category_map(dataset_root: Path) -> dict[str, int]:
    """Parse catidlist.txt and return a {category_name: category_id} mapping.

    Parameters
    ----------
    dataset_root : Path
        Root directory of the dataset (contains catidlist.txt).

    Returns
    -------
    dict[str, int]
    """
    cat_file = dataset_root / "catidlist.txt"
    if not cat_file.is_file():
        raise FileNotFoundError(f"Category list not found: {cat_file}")

    cat_names = np.genfromtxt(cat_file, delimiter=",", dtype=str, usecols=[0])
    cat_ids = np.genfromtxt(cat_file, delimiter=",", dtype=int, usecols=[1])
    return dict(zip(cat_names.tolist(), cat_ids.tolist()))


def extract_feature_vectors(
    root_dir: Path,
    dataset: str,
    descriptor: str = "sift",
) -> None:
    """Write per-category descriptor matrices as tab-delimited .tab files.

    Parameters
    ----------
    root_dir : Path
        Parent directory containing all datasets.
    dataset : str
        Dataset subdirectory name (e.g. 'VOC2006', 'Caltech101').
    descriptor : str, default 'sift'
        Descriptor type; determines the expected feature dimensionality.
    """
    dim = DESCRIPTOR_DIM.get(descriptor)
    if dim is None:
        raise ValueError(f"Unsupported descriptor '{descriptor}'. Choose from: {list(DESCRIPTOR_DIM)}")

    dataset_root = root_dir / dataset
    feature_dir = dataset_root / "FeatureMatrix"
    output_dir = dataset_root / "Fuzzy" / "Feature"
    output_dir.mkdir(parents=True, exist_ok=True)

    cat_map = _load_category_map(dataset_root)
    logger.info("Found %d categories in %s", len(cat_map), dataset)

    for cat_name in sorted(cat_map):
        src = feature_dir / f"{cat_name}.{descriptor}"
        if not src.is_file():
            logger.warning("Descriptor file not found, skipping: %s", src)
            continue

        logger.info("Reading %s ...", cat_name)
        col_range = np.arange(_DESCRIPTOR_OFFSET, _DESCRIPTOR_OFFSET + dim)
        cat_data = np.loadtxt(src, dtype=np.int16, usecols=col_range)

        dst = output_dir / f"{cat_name}.tab"
        np.savetxt(dst, cat_data, fmt="%d", delimiter="\t")
        logger.info("  → %s  (%d vectors)", dst, len(cat_data))


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Collect per-category SIFT/SURF descriptors into .tab files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--root-dir", required=True, type=Path, help="Data root directory")
    p.add_argument("--dataset", default="VOC2006", help="Dataset name")
    p.add_argument("--descriptor", default="sift", choices=list(DESCRIPTOR_DIM), help="Descriptor type")
    p.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    extract_feature_vectors(args.root_dir, args.dataset, args.descriptor)
    return 0


if __name__ == "__main__":
    sys.exit(main())
