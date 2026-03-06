"""
tests/test_classification.py — Unit tests for fuzzy_classify and cross_validate.

Tests verify return types, key presence, value ranges, and basic correctness
on a linearly separable synthetic dataset.
"""

import numpy as np
import pytest

from fuzzy_visual_encoding.classification import cross_validate, fuzzy_classify
from fuzzy_visual_encoding.dictionary import FuzzyDictionary


@pytest.fixture
def encoded_data() -> tuple[np.ndarray, np.ndarray]:
    """Small pre-encoded (N=120, 8-cluster) dataset with 3 balanced classes."""
    rng = np.random.default_rng(42)
    # Simulate three well-separated clusters in encoding space
    Z0 = rng.normal(loc=[1, 0, 0, 0, 0, 0, 0, 0], scale=0.1, size=(40, 8))
    Z1 = rng.normal(loc=[0, 1, 0, 0, 0, 0, 0, 0], scale=0.1, size=(40, 8))
    Z2 = rng.normal(loc=[0, 0, 1, 0, 0, 0, 0, 0], scale=0.1, size=(40, 8))
    Z = np.vstack([Z0, Z1, Z2])
    Z = np.clip(Z, 0, None)
    Z /= Z.sum(axis=1, keepdims=True) + 1e-12  # row-normalise
    y = np.array([0] * 40 + [1] * 40 + [2] * 40)
    return Z, y


class TestFuzzyClassify:
    def test_returns_dict_with_expected_keys(self, encoded_data):
        Z, y = encoded_data
        scores = fuzzy_classify(Z[:90], y[:90], Z[90:], y[90:])
        assert set(scores) == {"f1", "precision", "recall"}

    def test_scores_in_unit_range(self, encoded_data):
        Z, y = encoded_data
        scores = fuzzy_classify(Z[:90], y[:90], Z[90:], y[90:])
        for key, val in scores.items():
            assert 0.0 <= val <= 1.0, f"{key}={val} out of range"

    def test_separable_data_high_f1(self, encoded_data):
        Z, y = encoded_data
        scores = fuzzy_classify(Z[:90], y[:90], Z[90:], y[90:])
        assert scores["f1"] > 0.85, f"Expected high F1 on separable data, got {scores['f1']:.3f}"


class TestCrossValidate:
    def test_returns_dict_with_expected_keys(self, encoded_data):
        Z, y = encoded_data
        scores = cross_validate(Z, y, n_folds=3)
        expected = {
            "f1_mean", "f1_std",
            "precision_mean", "precision_std",
            "recall_mean", "recall_std",
        }
        assert set(scores) == expected

    def test_scores_in_unit_range(self, encoded_data):
        Z, y = encoded_data
        scores = cross_validate(Z, y, n_folds=3)
        for key, val in scores.items():
            assert 0.0 <= val <= 1.0, f"{key}={val} out of range"

    def test_std_nonnegative(self, encoded_data):
        Z, y = encoded_data
        scores = cross_validate(Z, y, n_folds=3)
        assert scores["f1_std"] >= 0
        assert scores["precision_std"] >= 0
        assert scores["recall_std"] >= 0

    def test_separable_data_high_mean_f1(self, encoded_data):
        Z, y = encoded_data
        scores = cross_validate(Z, y, n_folds=5)
        assert scores["f1_mean"] > 0.85

    def test_reproducible(self, encoded_data):
        Z, y = encoded_data
        s1 = cross_validate(Z, y, n_folds=5, random_state=0)
        s2 = cross_validate(Z, y, n_folds=5, random_state=0)
        assert s1["f1_mean"] == pytest.approx(s2["f1_mean"])
