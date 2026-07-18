"""Ensemble models built on lucidml trees.

A random forest reduces the variance of a single deep decision tree by
averaging many de-correlated trees. Two sources of randomness de-correlate
them:

1. **Bootstrap sampling** — each tree trains on ``n`` samples drawn *with
   replacement* from the training set.
2. **Random feature subsets** — at every split, only ``max_features``
   randomly chosen features compete for the best split.

Prediction is a majority vote over the trees. Because each tree overfits
in a *different* place (its own sliver-shaped artefacts land at different
locations), the vote averages the idiosyncrasies away while keeping the
signal the trees agree on.
"""

import numpy as np

from lucidml.trees import DecisionTreeClassifier

__all__ = ["RandomForestClassifier"]


class RandomForestClassifier:
    """Random forest: bagged decision trees with random feature subsets.

    Parameters
    ----------
    n_estimators : int, default=100
        Number of trees.
    max_depth : int or None, default=None
        Depth cap applied to every tree. ``None`` grows full trees — the
        standard choice for forests, since the vote handles the variance.
    min_samples_split : int, default=2
        Passed through to every tree. Must be at least 2.
    max_features : None, "sqrt" or int, default="sqrt"
        Features considered at each split of each tree. ``"sqrt"`` is the
        usual heuristic; ``None`` disables feature subsampling (pure
        bagging).
    impurity : {"entropy", "gini", "classification_error"} or callable, \
default="entropy"
        Impurity measure passed through to every tree.
    bootstrap : bool, default=True
        Draw a bootstrap sample (size ``n``, with replacement) for each
        tree. ``False`` trains every tree on the full training set, so
        randomness comes from feature subsets alone.
    random_state : int or None, default=None
        Seed controlling bootstrap draws and each tree's feature
        sampling. A fixed seed makes the whole forest reproducible.

    Attributes
    ----------
    estimators_ : list of DecisionTreeClassifier
        The fitted trees.
    classes_ : ndarray
        Sorted unique class labels seen during :meth:`fit`.
    n_features_ : int
        Number of features seen during :meth:`fit`.
    feature_importances_ : ndarray of shape (n_features,)
        Mean Decrease Impurity importances accumulated over all trees
        (property; available after fit).
    """

    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2,
                 max_features="sqrt", impurity="entropy", bootstrap=True,
                 random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.impurity = impurity
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.estimators_ = []

    def fit(self, X, y):
        """Fit ``n_estimators`` trees on bootstrap samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features.
        y : array-like of shape (n_samples,)
            Class labels.

        Returns
        -------
        RandomForestClassifier
            The fitted forest (``self``).

        Raises
        ------
        ValueError
            If ``n_estimators`` is not a positive integer, or if any
            tree-level parameter is invalid (raised by the trees).
        """
        if not (isinstance(self.n_estimators, (int, np.integer))
                and not isinstance(self.n_estimators, bool)
                and self.n_estimators >= 1):
            raise ValueError(
                f"n_estimators must be a positive integer, got {self.n_estimators!r}"
            )
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        n = X.shape[0]
        self.n_features_ = X.shape[1]
        self.classes_ = np.unique(y)
        rng = np.random.RandomState(self.random_state)

        self.estimators_ = []
        for _ in range(self.n_estimators):
            if self.bootstrap:
                idx = rng.randint(0, n, size=n)
            else:
                idx = np.arange(n)
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                impurity=self.impurity,
                max_features=self.max_features,
                random_state=rng.randint(0, 2**31 - 1),
            )
            tree.fit(X[idx], y[idx])
            self.estimators_.append(tree)
        return self

    def predict(self, X):
        """Majority vote over the trees' predictions.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to classify.

        Returns
        -------
        ndarray of shape (n_samples,)
            Predicted class label for each sample. Vote ties resolve to
            the smaller class label (deterministic).

        Raises
        ------
        RuntimeError
            If called before :meth:`fit`.
        """
        if not self.estimators_:
            raise RuntimeError("call fit before predict")
        X = np.asarray(X, dtype=float)
        # (n_estimators, n_samples) matrix of class indices
        votes = np.stack([
            np.searchsorted(self.classes_, tree.predict(X))
            for tree in self.estimators_
        ])
        counts = np.zeros((len(self.classes_), X.shape[0]), dtype=int)
        for k in range(len(self.classes_)):
            counts[k] = (votes == k).sum(axis=0)
        return self.classes_[counts.argmax(axis=0)]

    def score(self, X, y):
        """Mean accuracy of :meth:`predict` on ``(X, y)``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to classify.
        y : array-like of shape (n_samples,)
            True labels.

        Returns
        -------
        float
            Fraction of samples predicted correctly, in ``[0, 1]``.
        """
        y = np.asarray(y)
        return float(np.mean(self.predict(X) == y))

    @property
    def feature_importances_(self):
        """Mean Decrease Impurity importances over the whole forest.

        For every internal node of every tree, add ``w_n * gain_n`` to
        the node's split feature — ``w_n`` being the fraction of that
        tree's training samples reaching the node — then normalise once
        over the grand total, so the importances sum to 1.

        Returns
        -------
        ndarray of shape (n_features,)
            Forest-level importance of each feature. In-sample measure:
            computed from training-time gains, so noise features can
            receive non-zero importance and correlated features share
            (dilute) the credit of the signal they carry.

        Raises
        ------
        RuntimeError
            If called before :meth:`fit`.
        """
        if not self.estimators_:
            raise RuntimeError("call fit before feature_importances_")
        importances = np.zeros(self.n_features_)
        for tree in self.estimators_:
            n_root = tree.root.n_samples
            stack = [tree.root]
            while stack:
                node = stack.pop()
                if node.is_leaf:
                    continue
                importances[node.feature] += (node.n_samples / n_root) * node.gain
                stack.append(node.left)
                stack.append(node.right)
        total = importances.sum()
        return importances / total if total > 0 else importances
