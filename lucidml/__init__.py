"""lucidml — transparent, from-scratch machine-learning building blocks.

Readable NumPy implementations of core ML models, built to be understood,
tested, and extended. No black boxes.

Modules
-------
trees
    Decision-tree models: impurity measures, information gain, the
    best-split search, and :class:`~lucidml.trees.DecisionTreeClassifier`.

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

__version__ = "0.1.0"

__all__ = [
    "Node",
    "DecisionTreeClassifier",
    "entropy",
    "gini",
    "classification_error",
    "information_gain",
    "best_split",
    "__version__",
]
