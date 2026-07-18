"""Unit tests for lucidml.trees."""

import numpy as np
import pytest

from lucidml import (
    entropy,
    gini,
    classification_error,
    information_gain,
    best_split,
    DecisionTreeClassifier,
)
from lucidml.trees import _best_split_fast, _IMPURITIES


# --------------------------------------------------------------------------- #
# Impurity measures
# --------------------------------------------------------------------------- #
def test_entropy_half_half_is_one():
    assert entropy(np.array([0, 0, 1, 1])) == pytest.approx(1.0)


def test_entropy_pure_is_zero():
    assert entropy(np.array([1, 1, 1])) == pytest.approx(0.0)


def test_entropy_empty_is_zero():
    assert entropy(np.array([], dtype=int)) == 0.0


def test_gini_half_half():
    assert gini(np.array([0, 0, 1, 1])) == pytest.approx(0.5)


def test_gini_pure_is_zero():
    assert gini(np.array([7, 7, 7])) == pytest.approx(0.0)


def test_classification_error():
    # three of class 1, one of class 0 -> error = 1 - 3/4 = 0.25
    assert classification_error(np.array([0, 1, 1, 1])) == pytest.approx(0.25)


# --------------------------------------------------------------------------- #
# Information gain
# --------------------------------------------------------------------------- #
def test_information_gain_perfect_split_is_maximal():
    y = np.array([0, 0, 1, 1])
    gain = information_gain(y, np.array([0, 0]), np.array([1, 1]))
    assert gain == pytest.approx(1.0)


def test_information_gain_useless_split_is_zero():
    y = np.array([0, 1, 0, 1])
    gain = information_gain(y, np.array([0, 1]), np.array([0, 1]))
    assert gain == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# Best split
# --------------------------------------------------------------------------- #
def test_best_split_finds_separating_threshold():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 1, 1])
    split = best_split(X, y)
    assert split["feature"] == 0
    assert 1.0 < split["threshold"] < 2.0
    assert split["gain"] > 0


def test_best_split_picks_informative_feature():
    # feature 0 is pure noise (constant); feature 1 separates the classes
    X = np.array([[5.0, 0.0], [5.0, 0.0], [5.0, 1.0], [5.0, 1.0]])
    y = np.array([0, 0, 1, 1])
    split = best_split(X, y)
    assert split["feature"] == 1


# --------------------------------------------------------------------------- #
# DecisionTreeClassifier
# --------------------------------------------------------------------------- #
def test_tree_fits_linearly_separable_data():
    rng = np.random.RandomState(0)
    X = np.vstack([rng.normal(-2, 0.3, size=(50, 2)),
                   rng.normal(+2, 0.3, size=(50, 2))])
    y = np.array([0] * 50 + [1] * 50)
    clf = DecisionTreeClassifier(max_depth=3).fit(X, y)
    assert clf.score(X, y) == pytest.approx(1.0)


def test_predict_shape_and_label_set():
    rng = np.random.RandomState(1)
    X = rng.normal(size=(20, 2))
    y = (X[:, 0] > 0).astype(int)
    clf = DecisionTreeClassifier(max_depth=2).fit(X, y)
    preds = clf.predict(X)
    assert preds.shape == (20,)
    assert set(np.unique(preds)).issubset({0, 1})


def test_max_depth_is_respected():
    rng = np.random.RandomState(2)
    X = rng.normal(size=(200, 2))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    clf = DecisionTreeClassifier(max_depth=2).fit(X, y)
    assert clf.get_depth() <= 2


def test_gini_and_entropy_both_work():
    rng = np.random.RandomState(3)
    X = rng.normal(size=(80, 2))
    y = (X[:, 1] > 0).astype(int)
    for imp in ("entropy", "gini", "classification_error"):
        clf = DecisionTreeClassifier(max_depth=4, impurity=imp).fit(X, y)
        assert clf.score(X, y) >= 0.9


def test_callable_impurity_supported():
    rng = np.random.RandomState(4)
    X = rng.normal(size=(60, 2))
    y = (X[:, 0] > 0).astype(int)
    clf = DecisionTreeClassifier(max_depth=3, impurity=gini).fit(X, y)
    assert clf.score(X, y) >= 0.9


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError):
        DecisionTreeClassifier().predict(np.array([[0.0]]))


def test_unknown_impurity_raises():
    with pytest.raises(ValueError):
        DecisionTreeClassifier(impurity="banana").fit(
            np.array([[0.0], [1.0]]), np.array([0, 1])
        )


def test_min_samples_split_below_two_raises():
    # values < 2 are degenerate no-ops (the guard len(y) < m can never
    # fire for m <= 1), so they must be rejected loudly instead
    X = np.array([[0.0], [1.0]])
    y = np.array([0, 1])
    for bad in (1, 0, -3):
        with pytest.raises(ValueError):
            DecisionTreeClassifier(min_samples_split=bad).fit(X, y)


def test_invalid_max_depth_raises():
    X = np.array([[0.0], [1.0]])
    y = np.array([0, 1])
    with pytest.raises(ValueError):
        DecisionTreeClassifier(max_depth=0).fit(X, y)
    # None and positive ints remain valid
    assert DecisionTreeClassifier(max_depth=None).fit(X, y).score(X, y) == 1.0
    assert DecisionTreeClassifier(max_depth=1).fit(X, y).score(X, y) == 1.0


# --------------------------------------------------------------------------- #
# Fast split equivalence, MDI, max_features
# --------------------------------------------------------------------------- #
def test_fast_split_matches_reference():
    # the vectorised search must find a split achieving exactly the
    # reference loop's maximal gain, for every impurity, on random data
    for seed in range(5):
        rng = np.random.RandomState(seed)
        X = rng.normal(size=(40, 4))
        y = rng.randint(0, 3, size=40)
        for name, fn in _IMPURITIES.items():
            ref = best_split(X, y, fn)
            fast = _best_split_fast(X, y, name)
            assert np.isclose(fast["gain"], ref["gain"], atol=1e-9)
            achieved = information_gain(
                y, y[fast["left_mask"]], y[fast["right_mask"]], fn
            )
            assert np.isclose(achieved, ref["gain"], atol=1e-9)


def test_mdi_sums_to_one_and_ranks_informative_feature():
    rng = np.random.RandomState(1)
    X = rng.normal(size=(300, 2))
    y = (X[:, 0] > 0).astype(int)  # only feature 0 matters
    clf = DecisionTreeClassifier(max_depth=4).fit(X, y)
    imp = clf.feature_importances_
    assert imp.shape == (2,)
    assert np.all(imp >= 0)
    assert np.isclose(imp.sum(), 1.0)
    assert imp[0] > 0.8


def test_mdi_single_leaf_tree_is_all_zeros():
    clf = DecisionTreeClassifier().fit(
        np.array([[0.0], [1.0], [2.0]]), np.array([1, 1, 1])
    )
    assert np.array_equal(clf.feature_importances_, np.zeros(1))


def test_mdi_before_fit_raises():
    with pytest.raises(RuntimeError):
        DecisionTreeClassifier().feature_importances_


def test_max_features_validation_and_reproducibility():
    rng = np.random.RandomState(2)
    X = rng.normal(size=(120, 5))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    for bad in (0, -2, "banana", 1.5):
        with pytest.raises(ValueError):
            DecisionTreeClassifier(max_features=bad).fit(X, y)
    a = DecisionTreeClassifier(max_features=2, random_state=5).fit(X, y).predict(X)
    b = DecisionTreeClassifier(max_features=2, random_state=5).fit(X, y).predict(X)
    assert np.array_equal(a, b)
