"""
classification.py — SVM classification and cross-validation utilities.

This module provides two functions that operate on *encoded* feature vectors
(the output of :class:`~fuzzy_visual_encoding.dictionary.FuzzyDictionary`):

:func:`fuzzy_classify`
    Fits a one-vs-one RBF SVM on the training set and evaluates on a held-out
    test set.  Mirrors the original MATLAB pipeline in
    ``compFuzzyClassPerf.m`` / ``CatFuzzyClass.m``.

:func:`cross_validate`
    Runs stratified *k*-fold cross-validation and returns per-fold and
    aggregated metrics.  Corresponds to ``compFuzzyClassPerfCrossVal.m``.

Both functions report macro-averaged precision, recall, and F1-score, which
are more informative than accuracy on the imbalanced benchmark datasets used
in the original paper (VOC2006, Caltech101, Scene-15).
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC


def fuzzy_classify(
    Z_train: np.ndarray,
    y_train: np.ndarray,
    Z_test: np.ndarray,
    y_test: np.ndarray,
    kernel: str = "rbf",
    C: float = 1.0,
    gamma: str | float = "scale",
) -> dict[str, float]:
    """Train an SVM on fuzzy encodings and evaluate on a test split.

    Parameters
    ----------
    Z_train : ndarray of shape (N_train, n_clusters)
        Fuzzy encoding vectors for training images.
    y_train : ndarray of shape (N_train,)
        Class labels for training images.
    Z_test : ndarray of shape (N_test, n_clusters)
        Fuzzy encoding vectors for test images.
    y_test : ndarray of shape (N_test,)
        Ground-truth class labels for test images.
    kernel : str, default 'rbf'
        SVM kernel type, passed directly to :class:`sklearn.svm.SVC`.
    C : float, default 1.0
        SVM regularisation parameter.
    gamma : str or float, default 'scale'
        Kernel coefficient for 'rbf', 'poly', and 'sigmoid' kernels.

    Returns
    -------
    scores : dict with keys
        * ``'f1'`` — macro F1-score on the test set.
        * ``'precision'`` — macro precision.
        * ``'recall'`` — macro recall.
    """
    cls = SVC(kernel=kernel, C=C, gamma=gamma, probability=False)
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
    C: float = 1.0,
    gamma: str | float = "scale",
    random_state: int = 42,
) -> dict[str, float]:
    """Stratified *k*-fold cross-validation on fuzzy encodings.

    Splits ``Z`` into *n_folds* stratified folds, trains an SVM on each
    train split, and evaluates on the held-out fold.  Returns mean and
    standard deviation of per-fold macro F1, precision, and recall.

    Parameters
    ----------
    Z : ndarray of shape (N, n_clusters)
        Fuzzy encoding matrix for the full dataset.
    y : ndarray of shape (N,)
        Class labels.
    n_folds : int, default 10
        Number of stratified folds.
    kernel : str, default 'rbf'
        SVM kernel.
    C : float, default 1.0
        SVM regularisation parameter.
    gamma : str or float, default 'scale'
        Kernel coefficient.
    random_state : int, default 42
        Seed for the fold splitter.

    Returns
    -------
    scores : dict with keys
        * ``'f1_mean'``, ``'f1_std'``
        * ``'precision_mean'``, ``'precision_std'``
        * ``'recall_mean'``, ``'recall_std'``
    """
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    cls = SVC(kernel=kernel, C=C, gamma=gamma)

    f1_arr, pre_arr, rec_arr = [], [], []
    for train_idx, test_idx in cv.split(Z, y):
        pred = cls.fit(Z[train_idx], y[train_idx]).predict(Z[test_idx])
        f1_arr.append(f1_score(y[test_idx], pred, average="macro", zero_division=0))
        pre_arr.append(precision_score(y[test_idx], pred, average="macro", zero_division=0))
        rec_arr.append(recall_score(y[test_idx], pred, average="macro", zero_division=0))

    return {
        "f1_mean": float(np.mean(f1_arr)),
        "f1_std": float(np.std(f1_arr)),
        "precision_mean": float(np.mean(pre_arr)),
        "precision_std": float(np.std(pre_arr)),
        "recall_mean": float(np.mean(rec_arr)),
        "recall_std": float(np.std(rec_arr)),
    }
