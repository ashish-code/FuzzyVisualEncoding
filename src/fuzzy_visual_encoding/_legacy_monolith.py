"""
Fuzzy Visual Encoding using Gustafson-Kessel and Gath-Geva Clustering.

Python port of the MATLAB fuzzy clustering pipeline:
    calcFuzzyDict.m     → FuzzyDictionary.fit()
    calcFuzzyCoeff.m    → FuzzyDictionary.encode()
    compFuzzyCoeff.m    → FuzzyDictionary.encode()
    CatFuzzyClass.m     → fuzzy_classify() pipeline

The core algorithms — Gustafson-Kessel (GK) and Gath-Geva (GG) — are
implemented from scratch, replacing the MATLAB FuzzyClusteringToolbox.

Key contribution
----------------
Standard Bag-of-Features uses k-means (Euclidean, spherical clusters).
Fuzzy encoding uses GK/GG clustering which learns per-cluster covariance
matrices, modeling the true ellipsoidal shape of visual feature distributions.
This produces soft, overlapping cluster assignments (fuzzy memberships) that
capture ambiguity not expressible by hard k-means assignment.

Pipeline
--------
1. Sample ~50k SIFT vectors from all categories
2. PCA reduce 128-d → 10-d
3. Range-normalise to [0, 1]
4. Fit GK/GG fuzzy clustering → dictionary (centres + covariance matrices)
5. For each image: extract SIFT, PCA-reduce, compute fuzzy memberships
6. Average memberships per image → fuzzy encoding vector (N_words,)
7. SVM classify on fuzzy vectors

Usage
-----
    from fuzzy_encoding import FuzzyDictionary, fuzzy_classify
    import numpy as np

    # X_train: (N, d) PCA-reduced features
    fd = FuzzyDictionary(n_clusters=64, method='gk', fuzziness=2.0)
    fd.fit(X_train)
    Z_train = fd.encode(X_train)   # (N, n_clusters)
    Z_test  = fd.encode(X_test)

    scores = fuzzy_classify(Z_train, y_train, Z_test, y_test)
    print(f"F1: {scores['f1']:.3f}")
"""

from __future__ import annotations

import argparse
import warnings
from typing import Literal

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

# ---------------------------------------------------------------------------
# Fuzzy C-Means (base / initialisation for GG)
# ---------------------------------------------------------------------------

def _fuzzy_cmeans(
    X: np.ndarray,
    n_clusters: int,
    m: float = 2.0,
    max_iter: int = 1000,
    tol: float = 1e-12,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fuzzy C-Means clustering.

    Returns
    -------
    centers : (c, d)
    U       : (c, N) fuzzy membership matrix
    """
    if rng is None:
        rng = np.random.default_rng(42)
    N, d = X.shape
    c = n_clusters

    # Random initialisation of memberships
    U = rng.dirichlet(np.ones(c), size=N).T  # (c, N)

    for _ in range(max_iter):
        # Update centres
        um = U ** m  # (c, N)
        centers = (um @ X) / um.sum(axis=1, keepdims=True)  # (c, d)

        # Update memberships
        dist = np.zeros((c, N))
        for i in range(c):
            diff = X - centers[i]  # (N, d)
            dist[i] = np.sum(diff ** 2, axis=1)  # (N,)
        dist = np.maximum(dist, 1e-300)

        U_new = np.zeros_like(U)
        for i in range(c):
            ratios = (dist[i:i+1, :] / dist) ** (1.0 / (m - 1))  # (c, N)
            U_new[i] = 1.0 / ratios.sum(axis=0)

        if np.max(np.abs(U_new - U)) < tol:
            U = U_new
            break
        U = U_new

    return centers, U


# ---------------------------------------------------------------------------
# Gustafson-Kessel Clustering
# ---------------------------------------------------------------------------

class GustafsonKessel:
    """
    Gustafson-Kessel fuzzy clustering with adaptive covariance matrices.

    Parameters
    ----------
    n_clusters  : number of clusters c
    fuzziness   : m — fuzziness exponent (default 2.0)
    max_iter    : maximum iterations
    tol         : convergence threshold on membership change
    rho         : cluster volume weights (c,), defaults to ones
    random_state: seed
    """

    def __init__(
        self,
        n_clusters: int = 16,
        fuzziness: float = 2.0,
        max_iter: int = 1000,
        tol: float = 1e-12,
        rho: np.ndarray | None = None,
        random_state: int = 42,
    ) -> None:
        self.n_clusters = n_clusters
        self.fuzziness = fuzziness
        self.max_iter = max_iter
        self.tol = tol
        self.rho = rho
        self.random_state = random_state

        self.centers_: np.ndarray | None = None
        self.covariances_: list[np.ndarray] | None = None
        self.U_: np.ndarray | None = None  # (c, N)

    def fit(self, X: np.ndarray) -> "GustafsonKessel":
        """Fit GK clustering on data X (N, d)."""
        rng = np.random.default_rng(self.random_state)
        N, d = X.shape
        c = self.n_clusters
        m = self.fuzziness
        rho = self.rho if self.rho is not None else np.ones(c)

        # Initialise with FCM
        centers, U = _fuzzy_cmeans(X, c, m=m, max_iter=100, rng=rng)

        for iteration in range(self.max_iter):
            um = U ** m  # (c, N)

            # Update covariance matrices
            covs = []
            for i in range(c):
                diff = X - centers[i]  # (N, d)
                weighted = um[i, :, None] * diff  # (N, d)
                Ci = (weighted.T @ diff) / um[i].sum()  # (d, d)
                Ci += np.eye(d) * 1e-8  # regularisation
                # GK: scale by det(Ci)^(1/d) * rho[i]
                det_Ci = max(np.linalg.det(Ci), 1e-300)
                Ai = (det_Ci ** (1.0 / d)) * rho[i] * np.linalg.inv(Ci)
                covs.append(Ai)

            # Update memberships using Mahalanobis distance
            dist = np.zeros((c, N))
            for i in range(c):
                diff = X - centers[i]  # (N, d)
                # dist_i(x) = sqrt(x^T A_i x)
                dist[i] = np.einsum("nd,dd,nd->n", diff, covs[i], diff)
            dist = np.maximum(dist, 1e-300)

            U_new = np.zeros_like(U)
            for i in range(c):
                ratios = (dist[i:i+1, :] / dist) ** (1.0 / (m - 1))
                U_new[i] = 1.0 / ratios.sum(axis=0)

            # Update centres
            um_new = U_new ** m
            centers_new = (um_new @ X) / um_new.sum(axis=1, keepdims=True)

            delta = np.max(np.abs(U_new - U))
            U = U_new
            centers = centers_new
            if delta < self.tol:
                break

        self.centers_ = centers
        self.covariances_ = covs
        self.U_ = U
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Compute fuzzy membership matrix for new data.

        Returns U : (N, c)  membership of each sample in each cluster.
        """
        if self.centers_ is None:
            raise RuntimeError("Call fit() first.")

        N = len(X)
        c = self.n_clusters
        m = self.fuzziness

        dist = np.zeros((c, N))
        for i in range(c):
            diff = X - self.centers_[i]
            dist[i] = np.einsum("nd,dd,nd->n", diff, self.covariances_[i], diff)
        dist = np.maximum(dist, 1e-300)

        U = np.zeros((c, N))
        for i in range(c):
            ratios = (dist[i:i+1, :] / dist) ** (1.0 / (m - 1))
            U[i] = 1.0 / ratios.sum(axis=0)

        return U.T  # (N, c)


# ---------------------------------------------------------------------------
# Gath-Geva Clustering
# ---------------------------------------------------------------------------

class GathGeva:
    """
    Gath-Geva fuzzy clustering — extends GK with density estimation.

    Uses FCM initialisation followed by GG refinement. GG models each
    cluster as a Gaussian with full covariance, using a probabilistic
    distance function based on the Gauss kernel.

    Parameters
    ----------
    n_clusters  : number of clusters c
    fuzziness   : m — fuzziness exponent
    max_iter    : maximum iterations
    tol         : convergence threshold
    random_state: seed
    """

    def __init__(
        self,
        n_clusters: int = 16,
        fuzziness: float = 2.0,
        max_iter: int = 1000,
        tol: float = 1e-12,
        random_state: int = 42,
    ) -> None:
        self.n_clusters = n_clusters
        self.fuzziness = fuzziness
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

        self.centers_: np.ndarray | None = None
        self.covariances_: list[np.ndarray] | None = None
        self.priors_: np.ndarray | None = None
        self.U_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "GathGeva":
        """Fit GG clustering on data X (N, d)."""
        rng = np.random.default_rng(self.random_state)
        N, d = X.shape
        c = self.n_clusters
        m = self.fuzziness

        # FCM initialisation
        centers, U = _fuzzy_cmeans(X, c, m=m, max_iter=100, rng=rng)

        for iteration in range(self.max_iter):
            um = U ** m  # (c, N)
            priors = um.sum(axis=1) / N  # (c,)

            # Update covariance matrices (full Gaussian)
            covs = []
            for i in range(c):
                diff = X - centers[i]
                weighted = um[i, :, None] * diff
                Ci = (weighted.T @ diff) / um[i].sum()
                Ci += np.eye(d) * 1e-8
                covs.append(Ci)

            # GG probabilistic distance: d_i(x) = det(Ci)^(1/2) / pi_i
            #   * exp(0.5 * (x - v_i)^T Ci^{-1} (x - v_i))
            dist = np.zeros((c, N))
            for i in range(c):
                diff = X - centers[i]
                inv_Ci = np.linalg.inv(covs[i])
                maha = np.einsum("nd,dd,nd->n", diff, inv_Ci, diff)
                det_Ci = max(np.linalg.det(covs[i]), 1e-300)
                dist[i] = (det_Ci ** 0.5) / max(priors[i], 1e-300) * np.exp(0.5 * maha)
            dist = np.maximum(dist, 1e-300)

            U_new = np.zeros_like(U)
            for i in range(c):
                ratios = (dist[i:i+1, :] / dist) ** (1.0 / (m - 1))
                U_new[i] = 1.0 / ratios.sum(axis=0)

            um_new = U_new ** m
            centers_new = (um_new @ X) / um_new.sum(axis=1, keepdims=True)

            delta = np.max(np.abs(U_new - U))
            U = U_new
            centers = centers_new
            if delta < self.tol:
                break

        self.centers_ = centers
        self.covariances_ = covs
        self.priors_ = priors
        self.U_ = U
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Compute fuzzy memberships (N, c)."""
        if self.centers_ is None:
            raise RuntimeError("Call fit() first.")

        N = len(X)
        c = self.n_clusters
        m = self.fuzziness

        dist = np.zeros((c, N))
        for i in range(c):
            diff = X - self.centers_[i]
            inv_Ci = np.linalg.inv(self.covariances_[i])
            maha = np.einsum("nd,dd,nd->n", diff, inv_Ci, diff)
            det_Ci = max(np.linalg.det(self.covariances_[i]), 1e-300)
            dist[i] = (det_Ci ** 0.5) / max(self.priors_[i], 1e-300) * np.exp(0.5 * maha)
        dist = np.maximum(dist, 1e-300)

        U = np.zeros((c, N))
        for i in range(c):
            ratios = (dist[i:i+1, :] / dist) ** (1.0 / (m - 1))
            U[i] = 1.0 / ratios.sum(axis=0)

        return U.T  # (N, c)


# ---------------------------------------------------------------------------
# FuzzyDictionary — main encoding interface (≡ calcFuzzyDict.m + calcFuzzyCoeff.m)
# ---------------------------------------------------------------------------

Method = Literal["kmeans", "fcm", "gk", "gg"]


class FuzzyDictionary:
    """
    Fuzzy visual dictionary for Bag-of-Features encoding.

    Architecture
    ------------
    1. Sample feature vectors from all categories
    2. PCA-reduce to n_pca dimensions (default 10)
    3. Range-normalise to [0, 1]
    4. Fit fuzzy clustering (GK or GG) → dictionary
    5. Encode images as mean fuzzy membership vectors

    Parameters
    ----------
    n_clusters   : dictionary size (number of visual words)
    method       : 'kmeans' | 'fcm' | 'gk' | 'gg'
    n_pca        : PCA dimensionality (set None to skip PCA)
    fuzziness    : m parameter for FCM/GK/GG
    max_iter     : max clustering iterations
    random_state : seed
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
        self._scaler = MinMaxScaler()
        self._clusterer = None

    def _make_clusterer(self):
        if self.method == "gk":
            return GustafsonKessel(
                n_clusters=self.n_clusters,
                fuzziness=self.fuzziness,
                max_iter=self.max_iter,
                random_state=self.random_state,
            )
        elif self.method == "gg":
            return GathGeva(
                n_clusters=self.n_clusters,
                fuzziness=self.fuzziness,
                max_iter=self.max_iter,
                random_state=self.random_state,
            )
        elif self.method in ("kmeans", "fcm"):
            from sklearn.cluster import KMeans
            return KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=10)
        else:
            raise ValueError(f"Unknown method: {self.method!r}")

    def _preprocess(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
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

    def fit(self, X: np.ndarray) -> "FuzzyDictionary":
        """Learn fuzzy dictionary from data X (N, d)."""
        print(f"Fitting {self.method.upper()} dictionary ({self.n_clusters} clusters) on {X.shape} ...")
        Xp = self._preprocess(X, fit=True)
        self._clusterer = self._make_clusterer()
        self._clusterer.fit(Xp)
        return self

    def encode(self, X: np.ndarray) -> np.ndarray:
        """
        Compute fuzzy encoding for each row in X.

        For GK/GG: returns per-sample fuzzy membership vector (N, n_clusters).
        For KMeans: returns soft assignment via negative distances.

        Returns
        -------
        Z : (N, n_clusters)  fuzzy encoding matrix
        """
        if self._clusterer is None:
            raise RuntimeError("Call fit() first.")
        Xp = self._preprocess(X, fit=False)

        if self.method in ("gk", "gg"):
            return self._clusterer.predict_proba(Xp)  # (N, c)
        else:
            # For KMeans: use soft inverse-distance membership
            from sklearn.metrics.pairwise import euclidean_distances
            dists = euclidean_distances(Xp, self._clusterer.cluster_centers_)  # (N, c)
            dists = np.maximum(dists, 1e-300)
            m = self.fuzziness
            U = np.zeros_like(dists)
            for i in range(dists.shape[1]):
                ratios = (dists[:, i:i+1] / dists) ** (2.0 / (m - 1))
                U[:, i] = 1.0 / ratios.sum(axis=1)
            return U

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).encode(X)


# ---------------------------------------------------------------------------
# Classification helper (≡ compFuzzyClassPerf.m / CatFuzzyClass.m)
# ---------------------------------------------------------------------------

def fuzzy_classify(
    Z_train: np.ndarray,
    y_train: np.ndarray,
    Z_test: np.ndarray,
    y_test: np.ndarray,
    kernel: str = "rbf",
) -> dict:
    """Train RBF SVM on fuzzy encodings and evaluate."""
    cls = SVC(kernel=kernel, probability=False)
    cls.fit(Z_train, y_train)
    pred = cls.predict(Z_test)
    return {
        "f1": f1_score(y_test, pred, average="macro", zero_division=0),
        "precision": precision_score(y_test, pred, average="macro", zero_division=0),
        "recall": recall_score(y_test, pred, average="macro", zero_division=0),
    }


def cross_validate(
    Z: np.ndarray,
    y: np.ndarray,
    n_folds: int = 10,
    kernel: str = "rbf",
) -> dict:
    """10-fold stratified CV with RBF SVM on fuzzy encodings."""
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    f1_arr, pre_arr, rec_arr = [], [], []
    cls = SVC(kernel=kernel)
    for train, test in cv.split(Z, y):
        pred = cls.fit(Z[train], y[train]).predict(Z[test])
        f1_arr.append(f1_score(y[test], pred, average="macro", zero_division=0))
        pre_arr.append(precision_score(y[test], pred, average="macro", zero_division=0))
        rec_arr.append(recall_score(y[test], pred, average="macro", zero_division=0))
    return {
        "f1_mean": float(np.mean(f1_arr)),
        "f1_std": float(np.std(f1_arr)),
        "precision_mean": float(np.mean(pre_arr)),
        "recall_mean": float(np.mean(rec_arr)),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fuzzy visual encoding using Gustafson-Kessel / Gath-Geva clustering"
    )
    parser.add_argument("data", help="Feature file (whitespace-delimited, last col = label)")
    parser.add_argument("--method", choices=["kmeans", "fcm", "gk", "gg"], default="gk",
                        help="Clustering method")
    parser.add_argument("--n-clusters", type=int, default=64, help="Dictionary size")
    parser.add_argument("--n-pca", type=int, default=10, help="PCA dims (0 = skip PCA)")
    parser.add_argument("--fuzziness", type=float, default=2.0, help="Fuzziness exponent m")
    parser.add_argument("--n-folds", type=int, default=10, help="CV folds")
    parser.add_argument("--out", help="Output file for encoded features")
    args = parser.parse_args()

    print(f"Loading: {args.data}")
    data = np.loadtxt(args.data, delimiter=" ")
    X, y = data[:, :-1], data[:, -1]
    print(f"  {X.shape[0]} samples × {X.shape[1]} features")

    fd = FuzzyDictionary(
        n_clusters=args.n_clusters,
        method=args.method,
        n_pca=args.n_pca if args.n_pca > 0 else None,
        fuzziness=args.fuzziness,
    )
    Z = fd.fit_transform(X)
    print(f"Encoded shape: {Z.shape}")

    scores = cross_validate(Z, y, n_folds=args.n_folds)
    print(
        f"10-fold CV → F1: {scores['f1_mean']:.3f} ± {scores['f1_std']:.3f}  "
        f"| P: {scores['precision_mean']:.3f}  | R: {scores['recall_mean']:.3f}"
    )

    out = args.out or (args.data + f".{args.method}")
    np.savetxt(out, np.hstack([Z, y.reshape(-1, 1)]), fmt="%.6f", delimiter=" ")
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
