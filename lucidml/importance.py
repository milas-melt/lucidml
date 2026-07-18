"""Model-agnostic feature importance.

Permutation feature importance (PFI) measures how much a fitted model's
performance drops when one feature (or one *group* of features) is
randomly shuffled in the evaluation data. Unlike Mean Decrease Impurity
it is **out-of-sample** — evaluate it on held-out data and it cannot
credit features the model merely used to overfit — and it works for any
model exposing the scoring interface, not just trees.

Caveat: when features are correlated, shuffling one of them does not
remove its information (the model can still read it off the others), so
individually-permuted correlated features come out *under*-estimated.
The ``groups`` parameter addresses this by shuffling a whole group of
correlated columns jointly — importance at the group level.
"""

from types import SimpleNamespace

import numpy as np

__all__ = ["permutation_importance"]


def permutation_importance(model, X, y, n_repeats=10, random_state=None,
                           scoring=None, groups=None):
    """Permutation feature importance (PFI) of a fitted model.

    Computes the reference score ``s`` of ``model`` on ``(X, y)``, then
    for each feature (or feature group) and each of ``n_repeats``
    repetitions, shuffles that feature's column(s) and records the score
    drop ``s - s_shuffled``. Large positive drops mean the model relies
    on that feature; drops near zero mean it does not.

    Parameters
    ----------
    model : object
        Fitted model. Must expose ``score(X, y)`` unless a custom
        ``scoring`` callable is given. The model is **not** refitted.
    X : array-like of shape (n_samples, n_features)
        Evaluation features — use held-out data for an out-of-sample
        measure.
    y : array-like of shape (n_samples,)
        True labels for ``X``.
    n_repeats : int, default=10
        Number of independent shuffles per feature (group). More repeats
        tighten the standard deviation of the estimate.
    random_state : int or None, default=None
        Seed for the shuffles; fixes the result exactly.
    scoring : callable, optional
        ``scoring(model, X, y) -> float``, higher is better. Defaults to
        ``model.score``.
    groups : list of array-like of int, optional
        Feature-index groups to shuffle *jointly* (one permutation of the
        rows applied to all columns of the group, preserving the
        within-group joint distribution). ``None`` treats every feature
        as its own group. Use this for cluster-level importance of
        correlated features.

    Returns
    -------
    types.SimpleNamespace
        With fields:

        ``importances`` : ndarray of shape (n_groups, n_repeats)
            Score drop of every repetition.
        ``importances_mean`` : ndarray of shape (n_groups,)
            Mean drop per feature (group).
        ``importances_std`` : ndarray of shape (n_groups,)
            Standard deviation of the drop per feature (group).
        ``groups`` : list of list of int
            The groups actually used (each a list of column indices), in
            the order the result rows follow.

    Raises
    ------
    ValueError
        If ``n_repeats`` is not a positive integer or a group is empty.
    """
    if not (isinstance(n_repeats, (int, np.integer))
            and not isinstance(n_repeats, bool) and n_repeats >= 1):
        raise ValueError(f"n_repeats must be a positive integer, got {n_repeats!r}")

    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    n = X.shape[0]

    if scoring is None:
        def scoring(m, X_, y_):
            return m.score(X_, y_)

    if groups is None:
        groups = [[j] for j in range(X.shape[1])]
    groups = [list(np.atleast_1d(g).astype(int)) for g in groups]
    if any(len(g) == 0 for g in groups):
        raise ValueError("groups must be non-empty lists of column indices")

    rng = np.random.RandomState(random_state)
    reference = scoring(model, X, y)

    importances = np.zeros((len(groups), n_repeats))
    for gi, group in enumerate(groups):
        for k in range(n_repeats):
            perm = rng.permutation(n)
            X_shuffled = X.copy()
            X_shuffled[:, group] = X[perm][:, group]  # joint shuffle of the group
            importances[gi, k] = reference - scoring(model, X_shuffled, y)

    return SimpleNamespace(
        importances=importances,
        importances_mean=importances.mean(axis=1),
        importances_std=importances.std(axis=1),
        groups=groups,
    )
