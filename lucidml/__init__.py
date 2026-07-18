"""lucidml — transparent, from-scratch machine-learning building blocks.

Readable NumPy implementations of core ML models, built to be understood,
tested, and extended. No black boxes.

Modules
-------
trees
    Decision-tree models: impurity measures, information gain, the
    best-split search, and :class:`~lucidml.trees.DecisionTreeClassifier`
    with Mean Decrease Impurity feature importances.
ensemble
    :class:`~lucidml.ensemble.RandomForestClassifier` — bagged trees with
    per-split random feature subsets and forest-level MDI importances.
importance
    :func:`~lucidml.importance.permutation_importance` — model-agnostic,
    out-of-sample feature importance, with joint group shuffling for
    correlated features.

Examples
--------
>>> import numpy as np
>>> from lucidml import DecisionTreeClassifier
>>> X = np.array([[0.0], [1.0], [2.0], [3.0]])
>>> y = np.array([0, 0, 1, 1])
>>> DecisionTreeClassifier(max_depth=2).fit(X, y).score(X, y)
1.0
"""

from lucidml.trees import (
    Node,
    DecisionTreeClassifier,
    entropy,
    gini,
    classification_error,
    information_gain,
    best_split,
)
from lucidml.ensemble import RandomForestClassifier
from lucidml.importance import permutation_importance

__version__ = "0.2.0"

__all__ = [
    "Node",
    "DecisionTreeClassifier",
    "RandomForestClassifier",
    "permutation_importance",
    "entropy",
    "gini",
    "classification_error",
    "information_gain",
    "best_split",
    "__version__",
]
