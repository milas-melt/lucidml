"""Decision-tree models built from scratch on NumPy.

Everything in this module is plain NumPy — no scikit-learn — so every step
of the algorithm stays readable and easy to extend.

The building blocks are exposed directly:

- :func:`entropy`, :func:`gini`, :func:`classification_error` — impurity
  measures for a vector of class labels.
- :func:`information_gain` — impurity reduction achieved by a split.
- :func:`best_split` — exhaustive search for the best (feature, threshold)
  pair.
- :class:`Node` — one node of a fitted tree (internal node or leaf).
- :class:`DecisionTreeClassifier` — the estimator, with a
  scikit-learn-style ``fit`` / ``predict`` / ``score`` API.

Conventions
-----------
A split is the rule ``F_j <= tau``: samples satisfying it go to the *left*
child (region ``R_L``), the rest to the *right* child (region ``R_R``).
Candidate thresholds are the midpoints between consecutive unique values of
each feature, and the split retained is the one maximising the information
gain::

    IG(j, tau) = I(R) - |R_L|/|R| * I(R_L) - |R_R|/|R| * I(R_R)

where ``I`` is the impurity measure.
"""

import numpy as np

__all__ = [
    "entropy",
    "gini",
    "classification_error",
    "information_gain",
    "best_split",
    "Node",
    "DecisionTreeClassifier",
]


# --------------------------------------------------------------------------- #
# Impurity measures
# --------------------------------------------------------------------------- #
def class_probabilities(y):
    """Empirical class distribution of a label vector.

    Parameters
    ----------
    y : ndarray of shape (n_samples,)
        Class labels; any dtype accepted by :func:`numpy.unique`.

    Returns
    -------
    ndarray of shape (n_classes,)
        Fraction ``p_k`` of samples in each class present in ``y``, ordered
        as :func:`numpy.unique` sorts the labels. Fractions sum to 1.

    Examples
    --------
    >>> import numpy as np
    >>> class_probabilities(np.array([0, 0, 0, 1]))
    array([0.75, 0.25])
    """
    _, counts = np.unique(y, return_counts=True)
    return counts / counts.sum()


def entropy(y):
    """Shannon entropy (base 2) of a label vector.

    Defined as ``-sum_k p_k * log2(p_k)`` over the empirical class
    fractions ``p_k``. Zero for a pure node; maximal when classes are
    evenly represented (1 bit for a 50/50 binary node).

    Parameters
    ----------
    y : ndarray of shape (n_samples,)
        Class labels. May be empty.

    Returns
    -------
    float
        Entropy in bits; ``0.0`` for an empty or pure ``y``.

    Examples
    --------
    >>> import numpy as np
    >>> entropy(np.array([0, 0, 1, 1]))
    1.0
    >>> entropy(np.array([1, 1, 1]))
    0.0
    """
    if len(y) == 0:
        return 0.0
    p = class_probabilities(y)
    return float(-np.sum(p * np.log2(p)) + 0.0)  # +0.0 avoids a -0.0 display


def gini(y):
    """Gini impurity of a label vector.

    Defined as ``sum_k p_k * (1 - p_k)``: the probability of mislabelling a
    randomly drawn sample if labels were assigned by drawing from the
    node's class distribution. Zero for a pure node; 0.5 for a 50/50
    binary node.

    Parameters
    ----------
    y : ndarray of shape (n_samples,)
        Class labels. May be empty.

    Returns
    -------
    float
        Gini impurity in ``[0, 1)``; ``0.0`` for an empty or pure ``y``.

    Examples
    --------
    >>> import numpy as np
    >>> gini(np.array([0, 0, 1, 1]))
    0.5
    >>> gini(np.array([7, 7, 7]))
    0.0
    """
    if len(y) == 0:
        return 0.0
    p = class_probabilities(y)
    return float(np.sum(p * (1.0 - p)))


def classification_error(y):
    """Misclassification error of the majority-class rule.

    Defined as ``1 - max_k p_k``: the error made by predicting the most
    frequent class for every sample in the node.

    Parameters
    ----------
    y : ndarray of shape (n_samples,)
        Class labels. May be empty.

    Returns
    -------
    float
        Error rate in ``[0, 1)``; ``0.0`` for an empty or pure ``y``.

    Examples
    --------
    >>> import numpy as np
    >>> classification_error(np.array([0, 1, 1, 1]))
    0.25
    """
    if len(y) == 0:
        return 0.0
    p = class_probabilities(y)
    return float(1.0 - np.max(p))


# --------------------------------------------------------------------------- #
# Information gain and the best-split search
# --------------------------------------------------------------------------- #
def information_gain(y_parent, y_left, y_right, impurity=entropy):
    """Impurity reduction achieved by splitting a node into two children.

    Computes ``I(parent) - w_L * I(left) - w_R * I(right)`` where each
    weight is the fraction of parent samples routed to that child.

    Parameters
    ----------
    y_parent : ndarray of shape (n_samples,)
        Labels of the parent node.
    y_left, y_right : ndarray
        Labels of the two children; together they should partition
        ``y_parent`` (not checked).
    impurity : callable, default=entropy
        Impurity measure with signature ``f(y) -> float``.

    Returns
    -------
    float
        The information gain; ``0.0`` when the parent is empty.
        Non-negative for concave measures such as entropy and Gini.

    Examples
    --------
    A perfect split of a 50/50 parent removes one full bit of entropy:

    >>> import numpy as np
    >>> y = np.array([0, 0, 1, 1])
    >>> information_gain(y, np.array([0, 0]), np.array([1, 1]))
    1.0
    """
    n = len(y_parent)
    if n == 0:
        return 0.0
    w_left = len(y_left) / n
    w_right = len(y_right) / n
    return impurity(y_parent) - w_left * impurity(y_left) - w_right * impurity(y_right)


def best_split(X, y, impurity=entropy):
    """Exhaustively search for the split maximising information gain.

    For every feature ``j`` and every candidate threshold ``tau`` — the
    midpoints between consecutive unique values of that feature — evaluate
    the split ``F_j <= tau`` and keep the pair with the highest
    :func:`information_gain`.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Feature matrix of the node being split.
    y : ndarray of shape (n_samples,)
        Class labels of the node being split.
    impurity : callable, default=entropy
        Impurity measure passed through to :func:`information_gain`.

    Returns
    -------
    dict
        With keys:

        ``gain`` : float
            Best information gain found; ``-inf`` if no valid split exists.
        ``feature`` : int or None
            Index ``j`` of the splitting feature; ``None`` if no valid
            split exists (e.g. all feature columns are constant).
        ``threshold`` : float or None
            Threshold ``tau`` of the winning split.
        ``left_mask``, ``right_mask`` : ndarray of bool or None
            Boolean masks selecting each child's samples
            (``X[:, j] <= tau`` and its complement).

    Notes
    -----
    Complexity is ``O(n_features * n_thresholds * n_samples)`` per node —
    deliberately simple and readable rather than the sorted, incremental
    ``O(n log n)``-per-feature scan used by optimised libraries.

    Examples
    --------
    >>> import numpy as np
    >>> s = best_split(np.array([[0.0], [1.0], [2.0], [3.0]]),
    ...                np.array([0, 0, 1, 1]))
    >>> s["feature"], s["threshold"], s["gain"]
    (0, 1.5, 1.0)
    """
    best = {
        "gain": -np.inf,
        "feature": None,
        "threshold": None,
        "left_mask": None,
        "right_mask": None,
    }
    n_features = X.shape[1]
    for j in range(n_features):
        values = np.unique(X[:, j])
        if len(values) < 2:
            continue
        thresholds = (values[:-1] + values[1:]) / 2.0
        for tau in thresholds:
            left_mask = X[:, j] <= tau
            right_mask = ~left_mask
            if left_mask.sum() == 0 or right_mask.sum() == 0:
                continue
            gain = information_gain(y, y[left_mask], y[right_mask], impurity)
            if gain > best["gain"]:
                best = {
                    "gain": gain,
                    "feature": j,
                    "threshold": float(tau),
                    "left_mask": left_mask,
                    "right_mask": right_mask,
                }
    return best


# --------------------------------------------------------------------------- #
# Tree structure and estimator
# --------------------------------------------------------------------------- #
class Node:
    """One node of a fitted decision tree.

    A node plays one of two roles: an *internal node* carries a split
    ``(feature, threshold)`` and two children, while a *leaf* carries the
    class ``prediction`` for every sample that reaches it. The constructor
    is keyword-only to keep the two roles explicit at call sites.

    Parameters
    ----------
    feature : int, optional
        Index ``j`` of the splitting feature (internal nodes only).
    threshold : float, optional
        Threshold ``tau`` of the rule ``F_j <= tau`` (internal nodes only).
    left : Node, optional
        Child receiving samples with ``F_j <= tau`` (region ``R_L``).
    right : Node, optional
        Child receiving samples with ``F_j > tau`` (region ``R_R``).
    prediction : object, optional
        Class label stored by a leaf. A node with a non-``None``
        prediction is a leaf.
    """

    def __init__(self, *, feature=None, threshold=None, left=None, right=None, prediction=None):
        self.feature = feature        # feature index j used to split (internal)
        self.threshold = threshold    # threshold tau (internal)
        self.left = left              # child for  F_j <= tau  (region R_L)
        self.right = right            # child for  F_j >  tau  (region R_R)
        self.prediction = prediction  # class label (leaves only)

    @property
    def is_leaf(self):
        """bool : Whether this node is a leaf (stores a prediction)."""
        return self.prediction is not None


_IMPURITIES = {
    "entropy": entropy,
    "gini": gini,
    "classification_error": classification_error,
}


class DecisionTreeClassifier:
    """Binary-split decision tree grown by maximising information gain.

    The tree is built recursively: each node runs :func:`best_split` over
    all features and thresholds, splits while a split is worthwhile, and
    otherwise becomes a leaf storing its majority class.

    Parameters
    ----------
    max_depth : int or None, default=None
        Maximum depth of the tree. ``None`` grows nodes until another
        stopping rule fires.
    min_samples_split : int, default=2
        A node with fewer samples than this becomes a leaf.
    impurity : {"entropy", "gini", "classification_error"} or callable, \
default="entropy"
        Impurity measure used inside the information gain. A callable must
        have signature ``f(y) -> float``.

    Attributes
    ----------
    root : Node or None
        Root of the fitted tree; ``None`` until :meth:`fit` is called.

    Notes
    -----
    A node becomes a leaf when any stopping criterion fires: the node is
    pure, it holds fewer than ``min_samples_split`` samples, ``max_depth``
    is reached, or no candidate split yields positive information gain.
    The leaf stores the majority class of its samples.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[0.0], [1.0], [2.0], [3.0]])
    >>> y = np.array([0, 0, 1, 1])
    >>> clf = DecisionTreeClassifier(max_depth=2).fit(X, y)
    >>> clf.predict(np.array([[0.5], [2.5]]))
    array([0, 1])
    >>> clf.score(X, y)
    1.0
    >>> clf.get_depth()
    1
    """

    def __init__(self, max_depth=None, min_samples_split=2, impurity="entropy"):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.impurity = impurity
        self.root = None

    def _resolve_impurity(self):
        """Return the impurity callable selected by ``self.impurity``.

        Raises
        ------
        ValueError
            If ``self.impurity`` is a string naming no known measure.
        """
        if callable(self.impurity):
            return self.impurity
        try:
            return _IMPURITIES[self.impurity]
        except KeyError:
            raise ValueError(
                f"unknown impurity {self.impurity!r}; "
                f"choose from {list(_IMPURITIES)} or pass a callable"
            )

    def fit(self, X, y):
        """Grow the tree on training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features; converted to a float ndarray.
        y : array-like of shape (n_samples,)
            Class labels.

        Returns
        -------
        DecisionTreeClassifier
            The fitted estimator (``self``), enabling call chaining such
            as ``DecisionTreeClassifier().fit(X, y).predict(X)``.

        Raises
        ------
        ValueError
            If ``impurity`` names no known measure.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self._impurity_fn = self._resolve_impurity()
        self.root = self._build(X, y, depth=0)
        return self

    @staticmethod
    def _majority(y):
        """Most frequent label in ``y`` (ties go to the smallest label)."""
        vals, counts = np.unique(y, return_counts=True)
        return vals[np.argmax(counts)]

    def _build(self, X, y, depth):
        """Recursively grow and return the subtree for the samples ``(X, y)``.

        Applies the stopping criteria described in the class docstring;
        otherwise splits on :func:`best_split` and recurses one level
        deeper on each child.
        """
        # Stopping criteria -> create a leaf storing the majority class.
        if (
            len(np.unique(y)) == 1
            or len(y) < self.min_samples_split
            or (self.max_depth is not None and depth >= self.max_depth)
        ):
            return Node(prediction=self._majority(y))

        split = best_split(X, y, self._impurity_fn)
        if split["feature"] is None or split["gain"] <= 0:
            return Node(prediction=self._majority(y))

        left = self._build(X[split["left_mask"]], y[split["left_mask"]], depth + 1)
        right = self._build(X[split["right_mask"]], y[split["right_mask"]], depth + 1)
        return Node(
            feature=split["feature"],
            threshold=split["threshold"],
            left=left,
            right=right,
        )

    def _predict_one(self, x, node):
        """Route one sample down the tree; return its leaf's prediction."""
        while not node.is_leaf:
            node = node.left if x[node.feature] <= node.threshold else node.right
        return node.prediction

    def predict(self, X):
        """Predict a class label for each sample.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to classify; converted to a float ndarray.

        Returns
        -------
        ndarray of shape (n_samples,)
            Predicted class label for each sample.

        Raises
        ------
        RuntimeError
            If called before :meth:`fit`.
        """
        if self.root is None:
            raise RuntimeError("call fit before predict")
        X = np.asarray(X, dtype=float)
        return np.array([self._predict_one(x, self.root) for x in X])

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

    def get_depth(self):
        """Depth of the fitted tree.

        Returns
        -------
        int
            Number of edges on the longest root-to-leaf path. A tree that
            is a single leaf has depth 0; an unfitted tree also returns 0.
        """
        def _depth(node):
            if node is None or node.is_leaf:
                return 0
            return 1 + max(_depth(node.left), _depth(node.right))

        return _depth(self.root)
