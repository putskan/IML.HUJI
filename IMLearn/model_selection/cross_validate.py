from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float], cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """
    random_indexes = np.arange(len(X))
    np.random.shuffle(random_indexes)
    X, y = X[random_indexes], y[random_indexes]

    train_total_scores = 0
    validation_total_scores = 0
    original_estimator = estimator
    folds = np.array_split(np.arange(len(X)), cv)
    for fold in folds:
        train_mask = np.ones_like(y, dtype=bool)
        train_mask[fold] = False
        estimator = deepcopy(original_estimator).fit(X[train_mask], y[train_mask])
        train_predictions = estimator.predict(X[train_mask])
        val_predictions = estimator.predict(X[fold])
        train_total_scores += scoring(y[train_mask], train_predictions)
        validation_total_scores += scoring(y[fold], val_predictions)

    return train_total_scores / cv, validation_total_scores / cv
