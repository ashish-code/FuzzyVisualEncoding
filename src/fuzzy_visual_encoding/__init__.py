"""
fuzzy_visual_encoding
=====================

Fuzzy visual encoding for image classification.

This package implements Gustafson-Kessel (GK) and Gath-Geva (GG) fuzzy
clustering as drop-in replacements for k-means in the Bag-of-Features
pipeline, producing soft, overlapping cluster assignments that better capture
the ellipsoidal structure of visual feature distributions (e.g. SIFT, SURF).

Published in:
    A. Gupta et al., "Fuzzy Encoding for Visual Classification using
    Gustafson-Kessel Algorithm", IEEE ICIP 2012.

Submodules
----------
algorithms
    Low-level clustering implementations: GustafsonKessel, GathGeva, and
    the shared Fuzzy C-Means base (_fuzzy_cmeans).
dictionary
    FuzzyDictionary — the main Bag-of-Features interface (fit, encode,
    fit_transform), wrapping the clustering algorithms with optional PCA
    pre-processing and MinMax normalisation.
classification
    fuzzy_classify() and cross_validate() — SVM-based classification and
    stratified cross-validation utilities operating on encoded feature vectors.
cli
    Command-line entry point (``fuzzy-encode``) registered via pyproject.toml.

Quick start
-----------
>>> from fuzzy_visual_encoding import FuzzyDictionary, fuzzy_classify
>>> import numpy as np
>>> rng = np.random.default_rng(0)
>>> X = rng.standard_normal((500, 128))
>>> y = rng.integers(0, 5, 500)
>>> fd = FuzzyDictionary(n_clusters=32, method="gk")
>>> Z = fd.fit_transform(X)          # (500, 32) fuzzy encoding
>>> scores = fuzzy_classify(Z[:400], y[:400], Z[400:], y[400:])
>>> print(f"F1: {scores['f1']:.3f}")
"""

from fuzzy_visual_encoding.algorithms import GathGeva, GustafsonKessel
from fuzzy_visual_encoding.classification import cross_validate, fuzzy_classify
from fuzzy_visual_encoding.dictionary import FuzzyDictionary

__all__ = [
    "GustafsonKessel",
    "GathGeva",
    "FuzzyDictionary",
    "fuzzy_classify",
    "cross_validate",
]

__version__ = "0.1.0"
__author__ = "Ashish Gupta"
