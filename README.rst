======
bigwig
======


VERY EARLY STAGE Variable Importance Plot library


Description
===========

A very early stage, highly incomplete, proof-of-concept Python variable importance plotting library using single dispatch to mimic R's VIP library.

Usage
====
It's easy!

code::

  import numpy as np
  import matplotlib.pyplot as plt

  from sklearn import ensemble
  from sklearn import datasets
  from sklearn.utils import shuffle
  from sklearn.metrics import mean_squared_error

  import bigwig

  boston = datasets.load_boston()
  X, y = shuffle(boston.data, boston.target, random_state=13)
  X = X.astype(np.float32)
  offset = int(X.shape[0] * 0.9)

  X_train, y_train = X[:offset], y[:offset]
  X_test, y_test = X[offset:], y[offset:]

  params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
            'learning_rate': 0.01, 'loss': 'ls'}
  clf = ensemble.GradientBoostingRegressor(**params)
  clf.fit(X_train, y_train)

  # defaults:
  bigwig.vip(clf, boston.feature_names, relative=True, num_features=10, bar=True,
             horizontal=True, color="lightgrey", fill="lightgrey")

figure:: figure:: ./doc/images/default_example.png

code::
  # Vertical:
  bigwig.vip(clf, boston.feature_names, horizontal=False)

figure:: ./docs/images/non_default_example.png

Currently only with GB and RF regressions from Scikit Learn.

Note
====

This project has been set up using PyScaffold 3.1. For details and usage
information on PyScaffold see https://pyscaffold.org/.
