"""
tests/test_algorithms.py — Unit tests for GustafsonKessel, GathGeva, and FCM.

Tests verify:
    - Output shapes and membership normalisation (columns sum to 1)
    - Convergence on small synthetic data
    - Consistent results given a fixed random_state
    - predict_proba on held-out data produces valid memberships
    - RuntimeError raised if predict_proba called before fit
"""

import numpy as np
import pytest

from fuzzy_visual_encoding.algorithms import GathGeva, GustafsonKessel, _fuzzy_cmeans


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def blob_data() -> tuple[np.ndarray, np.ndarray]:
    """Two well-separated 2-d Gaussian blobs for deterministic clustering."""
    rng = np.random.default_rng(0)
    X0 = rng.normal(loc=[0.0, 0.0], scale=0.3, size=(50, 2))
    X1 = rng.normal(loc=[3.0, 3.0], scale=0.3, size=(50, 2))
    X = np.vstack([X0, X1])
    y = np.array([0] * 50 + [1] * 50)
    return X, y


# ---------------------------------------------------------------------------
# FCM (base)
# ---------------------------------------------------------------------------


class TestFuzzyCMeans:
    def test_output_shapes(self):
        rng = np.random.default_rng(1)
        X = rng.standard_normal((30, 4))
        centers, U = _fuzzy_cmeans(X, n_clusters=3, rng=rng)
        assert centers.shape == (3, 4)
        assert U.shape == (3, 30)

    def test_membership_sums_to_one(self):
        rng = np.random.default_rng(2)
        X = rng.standard_normal((40, 3))
        _, U = _fuzzy_cmeans(X, n_clusters=4, rng=rng)
        np.testing.assert_allclose(U.sum(axis=0), np.ones(40), atol=1e-9)

    def test_membership_nonnegative(self):
        rng = np.random.default_rng(3)
        X = rng.standard_normal((20, 2))
        _, U = _fuzzy_cmeans(X, n_clusters=3, rng=rng)
        assert (U >= 0).all()


# ---------------------------------------------------------------------------
# Gustafson-Kessel
# ---------------------------------------------------------------------------


class TestGustafsonKessel:
    def test_fit_returns_self(self, blob_data):
        X, _ = blob_data
        gk = GustafsonKessel(n_clusters=2, random_state=0)
        result = gk.fit(X)
        assert result is gk

    def test_fitted_attributes_populated(self, blob_data):
        X, _ = blob_data
        gk = GustafsonKessel(n_clusters=2, random_state=0).fit(X)
        assert gk.centers_ is not None
        assert gk.covariances_ is not None
        assert gk.U_ is not None

    def test_centers_shape(self, blob_data):
        X, _ = blob_data
        gk = GustafsonKessel(n_clusters=2, random_state=0).fit(X)
        assert gk.centers_.shape == (2, 2)

    def test_U_shape(self, blob_data):
        X, _ = blob_data
        gk = GustafsonKessel(n_clusters=2, random_state=0).fit(X)
        assert gk.U_.shape == (2, len(X))

    def test_U_columns_sum_to_one(self, blob_data):
        X, _ = blob_data
        gk = GustafsonKessel(n_clusters=2, random_state=0).fit(X)
        np.testing.assert_allclose(gk.U_.sum(axis=0), np.ones(len(X)), atol=1e-8)

    def test_predict_proba_shape(self, blob_data):
        X, _ = blob_data
        gk = GustafsonKessel(n_clusters=2, random_state=0).fit(X[:80])
        proba = gk.predict_proba(X[80:])
        assert proba.shape == (len(X[80:]), 2)

    def test_predict_proba_rows_sum_to_one(self, blob_data):
        X, _ = blob_data
        gk = GustafsonKessel(n_clusters=2, random_state=0).fit(X[:80])
        proba = gk.predict_proba(X[80:])
        np.testing.assert_allclose(proba.sum(axis=1), np.ones(len(X[80:])), atol=1e-8)

    def test_predict_proba_before_fit_raises(self):
        gk = GustafsonKessel(n_clusters=2)
        with pytest.raises(RuntimeError, match="fit\\(\\)"):
            gk.predict_proba(np.zeros((5, 2)))

    def test_cluster_separation(self, blob_data):
        """Dominant cluster assignment should match true labels for clean blobs."""
        X, y = blob_data
        gk = GustafsonKessel(n_clusters=2, random_state=0).fit(X)
        # Each observation should have >50% membership in one cluster
        dominant = gk.U_.argmax(axis=0)  # (N,)
        # The two dominant clusters should split cleanly between the two blobs
        assert len(np.unique(dominant[:50])) == 1
        assert len(np.unique(dominant[50:])) == 1
        assert dominant[0] != dominant[50]

    def test_reproducibility(self, blob_data):
        X, _ = blob_data
        gk1 = GustafsonKessel(n_clusters=2, random_state=7).fit(X)
        gk2 = GustafsonKessel(n_clusters=2, random_state=7).fit(X)
        np.testing.assert_array_equal(gk1.centers_, gk2.centers_)


# ---------------------------------------------------------------------------
# Gath-Geva
# ---------------------------------------------------------------------------


class TestGathGeva:
    def test_fit_returns_self(self, blob_data):
        X, _ = blob_data
        gg = GathGeva(n_clusters=2, random_state=0)
        assert gg.fit(X) is gg

    def test_fitted_attributes_populated(self, blob_data):
        X, _ = blob_data
        gg = GathGeva(n_clusters=2, random_state=0).fit(X)
        assert gg.centers_ is not None
        assert gg.priors_ is not None
        assert gg.covariances_ is not None

    def test_priors_shape(self, blob_data):
        X, _ = blob_data
        gg = GathGeva(n_clusters=2, random_state=0).fit(X)
        assert gg.priors_.shape == (2,)

    def test_predict_proba_shape(self, blob_data):
        X, _ = blob_data
        gg = GathGeva(n_clusters=2, random_state=0).fit(X[:80])
        proba = gg.predict_proba(X[80:])
        assert proba.shape == (len(X[80:]), 2)

    def test_predict_proba_rows_sum_to_one(self, blob_data):
        X, _ = blob_data
        gg = GathGeva(n_clusters=2, random_state=0).fit(X[:80])
        proba = gg.predict_proba(X[80:])
        np.testing.assert_allclose(proba.sum(axis=1), np.ones(len(X[80:])), atol=1e-8)

    def test_predict_proba_before_fit_raises(self):
        gg = GathGeva(n_clusters=2)
        with pytest.raises(RuntimeError, match="fit\\(\\)"):
            gg.predict_proba(np.zeros((5, 2)))
