"""Unit tests for lucidml.ensemble."""

import numpy as np
import pytest

from lucidml import DecisionTreeClassifier, RandomForestClassifier


def make_rings(n_per_class=150, noise=0.12, seed=0):
    """Two concentric rings with Gaussian noise (pure NumPy)."""
    rng = np.random.RandomState(seed)
    angles = rng.uniform(0, 2 * np.pi, size=2 * n_per_class)
    radii = np.r_[np.ones(n_per_class), 0.5 * np.ones(n_per_class)]
    X = np.c_[radii * np.cos(angles), radii * np.sin(angles)]
    X += rng.normal(scale=noise, size=X.shape)
    y = np.r_[np.zeros(n_per_class, dtype=int), np.ones(n_per_class, dtype=int)]
    shuffle = rng.permutation(len(y))
    return X[shuffle], y[shuffle]


def split_data(X, y, n_train):
    return X[:n_train], X[n_train:], y[:n_train], y[n_train:]


def test_forest_learns_nonlinear_data():
    X, y = make_rings()
    X_train, X_test, y_train, y_test = split_data(X, y, 210)
    forest = RandomForestClassifier(n_estimators=25, random_state=0)
    forest.fit(X_train, y_train)
    assert forest.score(X_test, y_test) >= 0.85


def test_forest_beats_single_tree_under_label_noise():
    # the variance claim: with noisy labels, averaging de-correlated trees
    # must beat one memorising tree out of sample
    X, y = make_rings(seed=1)
    X_train, X_test, y_train, y_test = split_data(X, y, 210)
    rng = np.random.RandomState(0)
    y_noisy = y_train.copy()
    flip = rng.choice(len(y_noisy), size=int(0.15 * len(y_noisy)), replace=False)
    y_noisy[flip] ^= 1

    tree = DecisionTreeClassifier().fit(X_train, y_noisy)
    forest = RandomForestClassifier(n_estimators=50, random_state=0)
    forest.fit(X_train, y_noisy)
    assert forest.score(X_test, y_test) > tree.score(X_test, y_test)


def test_forest_is_reproducible():
    X, y = make_rings(seed=2)
    f1 = RandomForestClassifier(n_estimators=10, random_state=7).fit(X, y)
    f2 = RandomForestClassifier(n_estimators=10, random_state=7).fit(X, y)
    assert np.array_equal(f1.predict(X), f2.predict(X))


def test_forest_mdi_shape_and_normalisation():
    X, y = make_rings(seed=3)
    forest = RandomForestClassifier(n_estimators=10, random_state=0).fit(X, y)
    imp = forest.feature_importances_
    assert imp.shape == (2,)
    assert imp.min() >= 0
    assert np.isclose(imp.sum(), 1.0)


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError):
        RandomForestClassifier().predict(np.array([[0.0, 0.0]]))


def test_bad_n_estimators_raises():
    X = np.array([[0.0], [1.0]])
    y = np.array([0, 1])
    for bad in (0, -1, 2.5):
        with pytest.raises(ValueError):
            RandomForestClassifier(n_estimators=bad).fit(X, y)


def test_bootstrap_false_still_fits():
    X, y = make_rings(seed=4)
    forest = RandomForestClassifier(n_estimators=5, bootstrap=False,
                                    random_state=0).fit(X, y)
    assert forest.score(X, y) > 0.9
