# lucidml

**Transparent, from-scratch machine-learning building blocks in NumPy.**

`lucidml` is a personal library of core machine-learning models implemented from
scratch — readable, dependency-light, and built to be understood, tested, and
extended. No black boxes: every model is plain NumPy you can step through.

**Module one:** decision trees.

## Install

```bash
cd lucidml
pip install -e ".[dev]"
```

The core library depends only on **NumPy**. The `[dev]` extra adds pytest,
scikit-learn, matplotlib and Jupyter for the tests and the example notebook.

## Quickstart

```python
import numpy as np
from lucidml import DecisionTreeClassifier

rng = np.random.RandomState(0)
X = rng.normal(size=(200, 2))
y = (X[:, 0] + X[:, 1] > 0).astype(int)

clf = DecisionTreeClassifier(max_depth=4, impurity="entropy").fit(X, y)
print("accuracy:", clf.score(X, y))
preds = clf.predict(X)
```

## What's inside

- `entropy`, `gini`, `classification_error` — impurity measures
- `information_gain` — gain of a candidate split
- `best_split` — exhaustive search over features and thresholds
- `Node`, `DecisionTreeClassifier` — the tree and its estimator API (`fit` / `predict` / `score` / `get_depth`)

See `examples/decision_tree_demo.ipynb` for a full walkthrough: non-linear ring
data, decision-boundary plots, and a Logistic Regression comparison.

## Tests

```bash
pytest
```

## Project layout

```
lucidml/
├── lucidml/            # the package
│   ├── __init__.py
│   └── trees.py
├── tests/              # pytest unit tests
│   └── test_trees.py
├── examples/
│   └── decision_tree_demo.ipynb
├── pyproject.toml
└── README.md
```

## Roadmap

- [ ] Regression trees (variance / MSE splitting)
- [ ] Cost-complexity pruning
- [ ] `RandomForest` ensemble
- [ ] Gradient-boosted trees
- [ ] Feature importance (MDI + permutation)

## License

MIT © Salim Tlemcani
