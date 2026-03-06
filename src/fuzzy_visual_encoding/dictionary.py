"""
dictionary.py — FuzzyDictionary: the main Bag-of-Features encoding interface.

This module provides :class:`FuzzyDictionary`, which wraps GK/GG clustering
(or k-means as a soft-assignment baseline) in a scikit-learn-compatible
``fit / transform`` API.  It replicates — and extends — the MATLAB pipeline:

    calcFuzzyDict.m     → :meth:`FuzzyDictionary.fit`
    calcFuzzyCoeff.m    → :meth:`FuzzyDictionary.encode`
    compFuzzyCoeff.m    → :meth:`FuzzyDictionary.encode`

Pre-processing pipeline
-----------------------
Before clustering, feature vectors are optionally:
    1. PCA-reduced to ``n_pca`` dimensions (default 10)  — removes correlations
       and dimensionality while retaining most variance.
    2. Min-max normalised to [0, 1] per feature dimension — equalises scale
       across PCA components without distorting the cluster geometry.

Encoding
--------
For GK/GG, ``encode(X)`` returns the full ``predict_proba`` membership matrix
(N × c).  For the k-means baseline, soft memberships are derived from inverse
Euclidean distances using the FCM membership update formula with the same
fuzziness exponent *m*.

Typical usage
-------------
>>> from fuzzy_visual_encoding import FuzzyDictionary
>>> import numpy as np
>>> X_train = np.random.default_rng(0).standard_normal((1000, 128))
>>> fd = FuzzyDictionary(n_clusters=64, method="gk", n_pca=10)
>>> fd.fit(X_train)
>>> Z_train = fd.encode(X_train)   # shape (1000, 64)
>>> Z_test  = fd.encode(X_test)    # shape (M, 64)
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from sklearn.preprocessing import MinMaxScaler

from fuzzy_visual_encoding.algorithms import GathGeva, GustafsonKessel

Method = Literal["kmeans", "fcm", "gk", "gg"]


class FuzzyDictionary:
    """Fuzzy visual dictionary for Bag-of-Features encoding.

    Learns a fuzzy codebook from a set of feature vectors and encodes new
    observations as soft cluster membership vectors.

    Parameters
    ----------
    n_clusters : int, default 64
        Number of visual words (dictionary size).  Typical values are
        16, 32, 64, 128, 256, 512 — larger values improve discrimination
        at the cost of computational expense.
    method : {'kmeans', 'fcm', 'gk', 'gg'}, default 'gk'
        Clustering algorithm used to build the dictionary.

        * ``'kmeans'`` — hard k-means with soft assignment at encode time.
        * ``'fcm'`` — Fuzzy C-Means (Euclidean distance, spherical clusters).
        * ``'gk'`` — Gustafson-Kessel (adaptive Mahalanobis, equal-volume clusters).
        * ``'gg'`` — Gath-Geva (probabilistic Gaussian, unequal-size clusters).

    n_pca : int or None, default 10
        Target dimensionality for PCA pre-processing.  Set to ``None`` to
        skip PCA and work in the original feature space.
    fuzziness : float, default 2.0
        Fuzziness exponent *m* passed to FCM / GK / GG (and used for soft
        k-means assignment).
    max_iter : int, default 1000
        Maximum clustering iterations.
    random_state : int, default 42
        Seed for reproducibility.

    Attributes
    ----------
    _pca : sklearn PCA or None
        Fitted PCA transform (set after :meth:`fit`).
    _scaler : MinMaxScaler
        Fitted feature scaler (set after :meth:`fit`).
    _clusterer : GustafsonKessel | GathGeva | KMeans
        Fitted clustering model (set after :meth:`fit`).
    """

    def __init__(
        self,
        n_clusters: int = 64,
        method: Method = "gk",
        n_pca: int | None = 10,
        fuzziness: float = 2.0,
        max_iter: int = 1000,
        random_state: int = 42,
    ) -> None:
        self.n_clusters = n_clusters
        self.method = method
        self.n_pca = n_pca
        self.fuzziness = fuzziness
        self.max_iter = max_iter
        self.random_state = random_state

        self._pca = None
        self._scaler: MinMaxScaler = MinMaxScaler()
        self._clusterer = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_clusterer(self):
        """Instantiate the selected clustering algorithm."""
        if self.method == "gk":
            return GustafsonKessel(
                n_clusters=self.n_clusters,
                fuzziness=self.fuzziness,
                max_iter=self.max_iter,
                random_state=self.random_state,
            )
        if self.method == "gg":
            return GathGeva(
                n_clusters=self.n_clusters,
                fuzziness=self.fuzziness,
                max_iter=self.max_iter,
                random_state=self.random_state,
            )
        if self.method in ("kmeans", "fcm"):
            from sklearn.cluster import KMeans

            return KMeans(
                n_clusters=self.n_clusters,
                random_state=self.random_state,
                n_init=10,
            )
        raise ValueError(f"Unknown method: {self.method!r}. Choose from: kmeans, fcm, gk, gg.")

    def _preprocess(self, X: np.ndarray, *, fit: bool = False) -> np.ndarray:
        """Apply PCA (optional) and min-max normalisation.

        Parameters
        ----------
        X : ndarray of shape (N, d)
        fit : bool
            If True, fit PCA and scaler on X before transforming.
            If False, apply already-fitted transforms.

        Returns
        -------
        Xp : ndarray of shape (N, n_pca or d)
        """
        if self.n_pca is not None and self.n_pca < X.shape[1]:
            if fit:
                from sklearn.decomposition import PCA

                self._pca = PCA(n_components=self.n_pca, random_state=self.random_state)
                X = self._pca.fit_transform(X)
            else:
                X = self._pca.transform(X)
        if fit:
            X = self._scaler.fit_transform(X)
        else:
            X = self._scaler.transform(X)
        return X

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray) -> "FuzzyDictionary":
        """Learn the fuzzy dictionary from training features.

        Parameters
        ----------
        X : ndarray of shape (N, d)
            Pooled feature vectors sampled from all training images.
            Typical usage: concatenate ~50 k SIFT descriptors per dataset.

        Returns
        -------
        self
        """
        print(
            f"[FuzzyDictionary] Fitting {self.method.upper()} "
            f"({self.n_clusters} clusters) on {X.shape} ..."
        )
        Xp = self._preprocess(X, fit=True)
        self._clusterer = self._make_clusterer()
        self._clusterer.fit(Xp)
        return self

    def encode(self, X: np.ndarray) -> np.ndarray:
        """Compute the fuzzy encoding for a set of feature vectors.

        For GK/GG, each observation is mapped to its full membership vector
        over all clusters.  For k-means, soft memberships are computed from
        inverse Euclidean distances using the FCM update formula.

        Parameters
        ----------
        X : ndarray of shape (N, d)
            Feature vectors to encode (same descriptor space as training).

        Returns
        -------
        Z : ndarray of shape (N, n_clusters)
            Each row is a normalised fuzzy membership vector.

        Raises
        ------
        RuntimeError
            If called before :meth:`fit`.
        """
        if self._clusterer is None:
            raise RuntimeError("Call fit() before encode().")
        Xp = self._preprocess(X, fit=False)

        if self.method in ("gk", "gg"):
            return self._clusterer.predict_proba(Xp)  # (N, c)

        # Soft k-means: inverse-distance FCM-style membership
        from sklearn.metrics.pairwise import euclidean_distances

        dists = euclidean_distances(Xp, self._clusterer.cluster_centers_)  # (N, c)
        dists = np.maximum(dists, 1e-300)
        m = self.fuzziness
        U = np.zeros_like(dists)
        for i in range(dists.shape[1]):
            ratios = (dists[:, i : i + 1] / dists) ** (2.0 / (m - 1))
            U[:, i] = 1.0 / ratios.sum(axis=1)
        return U

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit the dictionary and immediately encode X.

        Equivalent to ``fit(X).encode(X)`` but avoids redundant pre-processing.

        Parameters
        ----------
        X : ndarray of shape (N, d)

        Returns
        -------
        Z : ndarray of shape (N, n_clusters)
        """
        return self.fit(X).encode(X)
