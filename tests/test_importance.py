"""Unit tests for lucidml.importance."""

import numpy as np
import pytest

from lucidml import (
    DecisionTreeClassifier,
    RandomForestClassifier,
    permutation_importance,
)


def _informative_plus_noise(n=400, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.normal(size=(n, 2))
    y = (X[:, 0] > 0).astype(int)  # only feature 0 matters
    return X, y


def test_informative_feature_dominates_noise():
    X, y = _informative_plus_noise()
    model = DecisionTreeClassifier(max_depth=3).fit(X, y)
    result = permutation_importance(model, X, y, n_repeats=20, random_state=0)
    assert result.importances.shape == (2, 20)
    assert result.importances_mean[0] > 0.2
    assert abs(result.importances_mean[1]) < 0.05


def test_correlated_copies_underestimated_individually_fixed_by_groups():
    # duplicate the informative feature: a forest spreads its reliance over
    # both copies, so shuffling either copy alone understates its value;
    # shuffling the pair jointly reveals the true group importance
    rng = np.random.RandomState(1)
    base = rng.normal(size=(500, 1))
    X = np.hstack([base, base.copy(), rng.normal(size=(500, 1))])
    y = (base[:, 0] > 0).astype(int)
    model = RandomForestClassifier(n_estimators=30, max_features=1,
                                   random_state=0).fit(X, y)

    individual = permutation_importance(model, X, y, n_repeats=15, random_state=0)
    grouped = permutation_importance(model, X, y, n_repeats=15, random_state=0,
                                     groups=[[0, 1], [2]])

    group_drop = grouped.importances_mean[0]
    assert group_drop > individual.importances_mean[0]
    assert group_drop > individual.importances_mean[1]
    # the noise feature stays unimportant under both views
    assert abs(individual.importances_mean[2]) < 0.05
    assert abs(grouped.importances_mean[1]) < 0.05


def test_custom_scoring_is_used():
    X, y = _informative_plus_noise(seed=2)
    model = DecisionTreeClassifier(max_depth=3).fit(X, y)
    calls = []

    def scoring(m, X_, y_):
        calls.append(1)
        return m.score(X_, y_)

    permutation_importance(model, X, y, n_repeats=2, random_state=0,
                           scoring=scoring)
    # 1 reference call + 2 features * 2 repeats
    assert len(calls) == 1 + 2 * 2


def test_reproducible_with_seed():
    X, y = _informative_plus_noise(seed=3)
    model = DecisionTreeClassifier(max_depth=3).fit(X, y)
    r1 = permutation_importance(model, X, y, n_repeats=5, random_state=42)
    r2 = permutation_importance(model, X, y, n_repeats=5, random_state=42)
    assert np.array_equal(r1.importances, r2.importances)


def test_invalid_inputs_raise():
    X, y = _informative_plus_noise(seed=4)
    model = DecisionTreeClassifier(max_depth=2).fit(X, y)
    with pytest.raises(ValueError):
        permutation_importance(model, X, y, n_repeats=0)
    with pytest.raises(ValueError):
        permutation_importance(model, X, y, groups=[[0], []])
