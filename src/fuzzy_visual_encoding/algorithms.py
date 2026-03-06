"""
algorithms.py — Core fuzzy clustering algorithms.

Implements the two clustering methods that underpin the fuzzy visual encoding
pipeline, both derived from Fuzzy C-Means (FCM) but incorporating adaptive
covariance structure:

Gustafson-Kessel (GK)
    Extends FCM by assigning each cluster an adaptive *norm-inducing matrix*
    (proportional to the inverse local covariance).  The cluster shape adapts
    to the local data geometry, handling ellipsoidal clusters of equal volume
    but varying orientation and elongation.

    Reference:
        D. E. Gustafson and W. C. Kessel, "Fuzzy clustering with a fuzzy
        covariance matrix", CDC 1978.

Gath-Geva (GG)
    A further extension of GK that replaces the norm-based distance with a
    probabilistic distance derived from the Gaussian density function.  Each
    cluster's prior (mixing weight) is estimated from the fuzzy memberships,
    allowing clusters of unequal size and density.

    Reference:
        I. Gath and A. B. Geva, "Unsupervised optimal fuzzy clustering",
        IEEE TPAMI 1989.

Both estimators expose the same scikit-learn-style API::

    clusterer.fit(X)           → fits to (N, d) data
    clusterer.predict_proba(X) → returns (N, c) membership matrix

Internal dependency:  _fuzzy_cmeans() provides warm initialisation for both
GK and GG, improving convergence speed and stability.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Fuzzy C-Means — shared initialisation backbone
# ---------------------------------------------------------------------------


def _fuzzy_cmeans(
    X: np.ndarray,
    n_clusters: int,
    m: float = 2.0,
    max_iter: int = 1000,
    tol: float = 1e-12,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Fuzzy C-Means clustering used for warm-starting GK and GG.

    Updates cluster centres and a fuzzy membership matrix U by alternating
    between the two closed-form update equations derived from minimising the
    FCM objective (Bezdek 1981).

    Parameters
    ----------
    X : ndarray of shape (N, d)
        Input data — each row is one observation.
    n_clusters : int
        Number of clusters *c*.
    m : float, default 2.0
        Fuzziness exponent.  Values in [1.5, 3.0] are typical; m=1 degrades
        to hard k-means, m→∞ gives equal membership everywhere.
    max_iter : int, default 1000
        Maximum number of update iterations.
    tol : float, default 1e-12
        Convergence threshold on the maximum absolute change in U.
    rng : np.random.Generator or None
        Random number generator for reproducible initialisation.

    Returns
    -------
    centers : ndarray of shape (c, d)
        Cluster centre prototypes.
    U : ndarray of shape (c, N)
        Fuzzy membership matrix — U[i, n] is the degree of membership of
        observation n in cluster i.  Each column sums to 1.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    N, d = X.shape
    c = n_clusters

    # Initialise memberships via Dirichlet draw so each column sums to 1
    U = rng.dirichlet(np.ones(c), size=N).T  # (c, N)

    for _ in range(max_iter):
        um = U**m  # (c, N) — weighted membership matrix

        # Weighted mean → cluster centres
        centers = (um @ X) / um.sum(axis=1, keepdims=True)  # (c, d)

        # Squared Euclidean distances from each centre
        dist = np.zeros((c, N))
        for i in range(c):
            diff = X - centers[i]  # (N, d)
            dist[i] = np.sum(diff**2, axis=1)
        dist = np.maximum(dist, 1e-300)  # guard against zero division

        # Membership update (FCM Equation 2)
        U_new = np.zeros_like(U)
        for i in range(c):
            ratios = (dist[i : i + 1, :] / dist) ** (1.0 / (m - 1))  # (c, N)
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
    """Gustafson-Kessel fuzzy clustering with per-cluster adaptive covariances.

    GK replaces the Euclidean distance of standard FCM with a Mahalanobis
    distance whose norm-inducing matrix *A_i* adapts to the local covariance
    of each cluster:

        A_i = [det(C_i)]^{1/d} · ρ_i · C_i^{-1}

    where C_i is the fuzzy-weighted covariance and ρ_i is an optional volume
    weight (default 1).  Constraining the determinant to ρ_i means each
    cluster occupies equal hypervolume, preventing the *singularity collapse*
    that affects unconstrained adaptive-distance FCM.

    The algorithm alternates three update steps until convergence:
        1. Recompute weighted covariance matrices C_i
        2. Derive norm matrices A_i from C_i
        3. Recompute memberships using the Mahalanobis distance induced by A_i

    Parameters
    ----------
    n_clusters : int, default 16
        Number of clusters *c*.
    fuzziness : float, default 2.0
        Fuzziness exponent *m*.
    max_iter : int, default 1000
        Maximum number of GK iterations (after FCM warm-start).
    tol : float, default 1e-12
        Convergence threshold on max absolute membership change.
    rho : ndarray of shape (c,) or None
        Per-cluster volume weights.  None defaults to all-ones (equal volumes).
    random_state : int, default 42
        Seed for the internal random number generator.

    Attributes
    ----------
    centers_ : ndarray of shape (c, d)
        Cluster prototype vectors after fitting.
    covariances_ : list of ndarray, each of shape (d, d)
        Fitted norm matrices A_i (not raw covariances).
    U_ : ndarray of shape (c, N)
        Final fuzzy membership matrix on training data.
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
        self.U_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "GustafsonKessel":
        """Fit GK clustering to data.

        Parameters
        ----------
        X : ndarray of shape (N, d)
            Training observations.

        Returns
        -------
        self
        """
        rng = np.random.default_rng(self.random_state)
        N, d = X.shape
        c = self.n_clusters
        m = self.fuzziness
        rho = self.rho if self.rho is not None else np.ones(c)

        # Warm-start with a short FCM run
        centers, U = _fuzzy_cmeans(X, c, m=m, max_iter=100, rng=rng)

        for _ in range(self.max_iter):
            um = U**m  # (c, N)

            # Step 1 — Compute fuzzy-weighted covariances and GK norm matrices
            covs: list[np.ndarray] = []
            for i in range(c):
                diff = X - centers[i]  # (N, d)
                weighted = um[i, :, None] * diff  # (N, d)
                Ci = (weighted.T @ diff) / um[i].sum()  # (d, d)
                Ci += np.eye(d) * 1e-8  # Tikhonov regularisation for stability
                # GK constraint: volume controlled by det(C)^{1/d} · rho
                det_Ci = max(np.linalg.det(Ci), 1e-300)
                Ai = (det_Ci ** (1.0 / d)) * rho[i] * np.linalg.inv(Ci)
                covs.append(Ai)

            # Step 2 — Mahalanobis distances using norm matrices A_i
            dist = np.zeros((c, N))
            for i in range(c):
                diff = X - centers[i]  # (N, d)
                dist[i] = np.einsum("nd,dd,nd->n", diff, covs[i], diff)
            dist = np.maximum(dist, 1e-300)

            # Step 3 — Membership update
            U_new = np.zeros_like(U)
            for i in range(c):
                ratios = (dist[i : i + 1, :] / dist) ** (1.0 / (m - 1))
                U_new[i] = 1.0 / ratios.sum(axis=0)

            # Step 4 — Centre update
            um_new = U_new**m
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
        """Compute soft cluster memberships for new observations.

        Parameters
        ----------
        X : ndarray of shape (N, d)

        Returns
        -------
        U : ndarray of shape (N, c)
            U[n, i] is the membership degree of observation n in cluster i.
            Each row sums to 1.

        Raises
        ------
        RuntimeError
            If called before :meth:`fit`.
        """
        if self.centers_ is None:
            raise RuntimeError("Call fit() before predict_proba().")

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
            ratios = (dist[i : i + 1, :] / dist) ** (1.0 / (m - 1))
            U[i] = 1.0 / ratios.sum(axis=0)

        return U.T  # (N, c)


# ---------------------------------------------------------------------------
# Gath-Geva Clustering
# ---------------------------------------------------------------------------


class GathGeva:
    """Gath-Geva fuzzy clustering with probabilistic Gaussian distance.

    GG extends GK by replacing the Mahalanobis distance with a *probabilistic
    distance* derived from the multivariate Gaussian density:

        d_i(x) = [det(C_i)]^{1/2} / π_i · exp(½ · (x − v_i)ᵀ C_i^{-1} (x − v_i))

    where π_i = Σ_n u_{in}^m / N is the fuzzy prior of cluster i.  This
    formulation allows clusters of unequal size (prior) and density (det(C_i)),
    producing more flexible partitions than GK at the cost of requiring the
    priors to be well-estimated.

    Parameters
    ----------
    n_clusters : int, default 16
    fuzziness : float, default 2.0
    max_iter : int, default 1000
    tol : float, default 1e-12
    random_state : int, default 42

    Attributes
    ----------
    centers_ : ndarray of shape (c, d)
    covariances_ : list of ndarray, each (d, d) — raw cluster covariances C_i
    priors_ : ndarray of shape (c,) — fuzzy cluster priors π_i
    U_ : ndarray of shape (c, N)
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
        """Fit GG clustering to data.

        Parameters
        ----------
        X : ndarray of shape (N, d)

        Returns
        -------
        self
        """
        rng = np.random.default_rng(self.random_state)
        N, d = X.shape
        c = self.n_clusters
        m = self.fuzziness

        # Warm-start with FCM
        centers, U = _fuzzy_cmeans(X, c, m=m, max_iter=100, rng=rng)

        for _ in range(self.max_iter):
            um = U**m  # (c, N)
            priors = um.sum(axis=1) / N  # (c,) — cluster mixing weights

            # Full Gaussian covariances
            covs: list[np.ndarray] = []
            for i in range(c):
                diff = X - centers[i]
                weighted = um[i, :, None] * diff
                Ci = (weighted.T @ diff) / um[i].sum()
                Ci += np.eye(d) * 1e-8
                covs.append(Ci)

            # GG probabilistic distance:
            #   d_i(x) = sqrt(det(C_i)) / π_i * exp(0.5 * maha_i(x))
            # Clip maha before exp to prevent overflow → inf, which would cause
            # inf/inf = nan in the membership ratio step.
            dist = np.zeros((c, N))
            for i in range(c):
                diff = X - centers[i]
                inv_Ci = np.linalg.inv(covs[i])
                maha = np.einsum("nd,dd,nd->n", diff, inv_Ci, diff)
                det_Ci = max(np.linalg.det(covs[i]), 1e-300)
                dist[i] = (det_Ci**0.5) / max(priors[i], 1e-300) * np.exp(np.minimum(0.5 * maha, 500.0))
            dist = np.nan_to_num(dist, nan=1e300, posinf=1e300)
            dist = np.maximum(dist, 1e-300)

            U_new = np.zeros_like(U)
            for i in range(c):
                ratios = (dist[i : i + 1, :] / dist) ** (1.0 / (m - 1))
                U_new[i] = 1.0 / ratios.sum(axis=0)

            um_new = U_new**m
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
        """Compute soft cluster memberships for new observations.

        Parameters
        ----------
        X : ndarray of shape (N, d)

        Returns
        -------
        U : ndarray of shape (N, c)

        Raises
        ------
        RuntimeError
            If called before :meth:`fit`.
        """
        if self.centers_ is None:
            raise RuntimeError("Call fit() before predict_proba().")

        N = len(X)
        c = self.n_clusters
        m = self.fuzziness

        dist = np.zeros((c, N))
        for i in range(c):
            diff = X - self.centers_[i]
            inv_Ci = np.linalg.inv(self.covariances_[i])
            maha = np.einsum("nd,dd,nd->n", diff, inv_Ci, diff)
            det_Ci = max(np.linalg.det(self.covariances_[i]), 1e-300)
            dist[i] = (det_Ci**0.5) / max(self.priors_[i], 1e-300) * np.exp(np.minimum(0.5 * maha, 500.0))
        dist = np.nan_to_num(dist, nan=1e300, posinf=1e300)
        dist = np.maximum(dist, 1e-300)

        U = np.zeros((c, N))
        for i in range(c):
            ratios = (dist[i : i + 1, :] / dist) ** (1.0 / (m - 1))
            U[i] = 1.0 / ratios.sum(axis=0)

        return U.T  # (N, c)
