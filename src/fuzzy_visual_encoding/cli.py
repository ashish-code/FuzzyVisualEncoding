"""
cli.py — Command-line entry point for fuzzy visual encoding.

Registered as ``fuzzy-encode`` via the ``[project.scripts]`` table in
``pyproject.toml``.  After installation (``uv sync`` or ``pip install -e .``),
run::

    fuzzy-encode features.txt --method gk --n-clusters 64 --n-folds 10

The input file must be a whitespace-delimited text file where each row is one
observation and the **last column** is the integer class label.

Example
-------
    $ fuzzy-encode data/features.txt \\
          --method gk \\
          --n-clusters 64 \\
          --n-pca 10 \\
          --fuzziness 2.0 \\
          --n-folds 10 \\
          --out data/encoded_gk64.txt
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from fuzzy_visual_encoding.classification import cross_validate
from fuzzy_visual_encoding.dictionary import FuzzyDictionary


def _build_parser() -> argparse.ArgumentParser:
    """Construct and return the argument parser."""
    parser = argparse.ArgumentParser(
        prog="fuzzy-encode",
        description=(
            "Fuzzy visual encoding using Gustafson-Kessel (GK) or Gath-Geva (GG) "
            "clustering.  Reads a feature matrix from a text file, builds a fuzzy "
            "dictionary, encodes each observation as a soft membership vector, and "
            "reports stratified cross-validation performance."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "data",
        help="Feature file — whitespace-delimited, last column = integer class label.",
    )
    parser.add_argument(
        "--method",
        choices=["kmeans", "fcm", "gk", "gg"],
        default="gk",
        help="Clustering algorithm used to build the fuzzy dictionary.",
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=64,
        metavar="C",
        help="Number of visual words (dictionary size).",
    )
    parser.add_argument(
        "--n-pca",
        type=int,
        default=10,
        metavar="D",
        help="PCA dimensionality before clustering (0 = skip PCA).",
    )
    parser.add_argument(
        "--fuzziness",
        type=float,
        default=2.0,
        metavar="M",
        help="Fuzziness exponent m (1 < m; typically 1.5–3.0).",
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=10,
        metavar="K",
        help="Number of stratified cross-validation folds.",
    )
    parser.add_argument(
        "--out",
        metavar="FILE",
        help=(
            "Output path for the encoded feature matrix (tab-delimited, last "
            "column = label).  Defaults to <data>.<method>."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point called by the ``fuzzy-encode`` console script.

    Parameters
    ----------
    argv : list[str] or None
        Argument vector; defaults to ``sys.argv[1:]`` when None.

    Returns
    -------
    int
        Exit code (0 on success).
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    # ------------------------------------------------------------------ Load
    print(f"Loading: {args.data}")
    try:
        data = np.loadtxt(args.data)
    except OSError as exc:
        print(f"Error reading '{args.data}': {exc}", file=sys.stderr)
        return 1

    X, y = data[:, :-1], data[:, -1].astype(int)
    print(f"  {X.shape[0]:,} samples × {X.shape[1]} features  |  {len(np.unique(y))} classes")

    # --------------------------------------------------------------- Encode
    fd = FuzzyDictionary(
        n_clusters=args.n_clusters,
        method=args.method,
        n_pca=args.n_pca if args.n_pca > 0 else None,
        fuzziness=args.fuzziness,
    )
    Z = fd.fit_transform(X)
    print(f"Encoded shape: {Z.shape}")

    # ---------------------------------------------------------- Cross-validate
    scores = cross_validate(Z, y, n_folds=args.n_folds)
    print(
        f"\n{args.n_folds}-fold CV ({args.method.upper()}, C={args.n_clusters}):\n"
        f"  F1        = {scores['f1_mean']:.4f} ± {scores['f1_std']:.4f}\n"
        f"  Precision = {scores['precision_mean']:.4f} ± {scores['precision_std']:.4f}\n"
        f"  Recall    = {scores['recall_mean']:.4f} ± {scores['recall_std']:.4f}"
    )

    # ----------------------------------------------------------------- Save
    out_path = args.out or f"{args.data}.{args.method}"
    np.savetxt(
        out_path,
        np.hstack([Z, y.reshape(-1, 1)]),
        fmt="%.6f",
        delimiter="\t",
    )
    print(f"\nSaved encoded features → {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
