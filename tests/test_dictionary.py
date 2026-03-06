"""
tests/test_dictionary.py — Unit tests for FuzzyDictionary.

Tests cover:
    - fit / encode / fit_transform shapes for all four methods
    - Encoding values in [0, 1] and row-sums close to 1 for GK/GG
    - RuntimeError raised if encode called before fit
    - PCA dimensionality reduction path (n_pca < d)
    - n_pca=None path (no PCA)
    - Consistency across repeated calls
"""

import numpy as np
import pytest

from fuzzy_visual_encoding.dictionary import FuzzyDictionary


@pytest.fixture
def small_data() -> tuple[np.ndarray, np.ndarray]:
    """Small (N=200, d=16) dataset with 4 classes."""
    rng = np.random.default_rng(99)
    X = rng.standard_normal((200, 16))
    y = rng.integers(0, 4, 200)
    return X, y


@pytest.mark.parametrize("method", ["kmeans", "fcm", "gk", "gg"])
class TestFuzzyDictionaryMethods:
    def test_fit_transform_shape(self, method, small_data):
        X, _ = small_data
        fd = FuzzyDictionary(n_clusters=8, method=method, n_pca=4, random_state=0)
        Z = fd.fit_transform(X)
        assert Z.shape == (len(X), 8)

    def test_encode_shape(self, method, small_data):
        X, _ = small_data
        fd = FuzzyDictionary(n_clusters=8, method=method, n_pca=4, random_state=0)
        fd.fit(X[:150])
        Z = fd.encode(X[150:])
        assert Z.shape == (50, 8)

    def test_values_in_unit_range(self, method, small_data):
        X, _ = small_data
        fd = FuzzyDictionary(n_clusters=8, method=method, n_pca=4, random_state=0)
        Z = fd.fit_transform(X)
        assert Z.min() >= -1e-9
        assert Z.max() <= 1 + 1e-9

    def test_encode_before_fit_raises(self, method, small_data):
        X, _ = small_data
        fd = FuzzyDictionary(n_clusters=8, method=method)
        with pytest.raises(RuntimeError, match="fit\\(\\)"):
            fd.encode(X)


class TestFuzzyDictionaryOptions:
    def test_no_pca(self, small_data):
        X, _ = small_data
        fd = FuzzyDictionary(n_clusters=6, method="gk", n_pca=None, random_state=0)
        Z = fd.fit_transform(X)
        assert Z.shape == (len(X), 6)

    def test_pca_reduces_dimension(self, small_data):
        X, _ = small_data
        fd = FuzzyDictionary(n_clusters=6, method="gk", n_pca=4, random_state=0)
        fd.fit(X)
        assert fd._pca.n_components_ == 4

    def test_invalid_method_raises(self, small_data):
        X, _ = small_data
        with pytest.raises(ValueError, match="Unknown method"):
            FuzzyDictionary(n_clusters=4, method="invalid").fit(X)  # type: ignore

    def test_gk_memberships_sum_to_one(self, small_data):
        X, _ = small_data
        fd = FuzzyDictionary(n_clusters=8, method="gk", n_pca=4, random_state=0)
        Z = fd.fit_transform(X)
        np.testing.assert_allclose(Z.sum(axis=1), np.ones(len(X)), atol=1e-7)

    def test_gg_memberships_sum_to_one(self, small_data):
        X, _ = small_data
        fd = FuzzyDictionary(n_clusters=8, method="gg", n_pca=4, random_state=0)
        Z = fd.fit_transform(X)
        np.testing.assert_allclose(Z.sum(axis=1), np.ones(len(X)), atol=1e-7)

    def test_fit_transform_consistent_with_fit_encode(self, small_data):
        X, _ = small_data
        fd1 = FuzzyDictionary(n_clusters=8, method="gk", n_pca=4, random_state=0)
        Z1 = fd1.fit_transform(X)

        fd2 = FuzzyDictionary(n_clusters=8, method="gk", n_pca=4, random_state=0)
        fd2.fit(X)
        Z2 = fd2.encode(X)

        np.testing.assert_allclose(Z1, Z2, atol=1e-10)
