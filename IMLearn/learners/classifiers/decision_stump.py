from __future__ import annotations
from typing import Tuple, NoReturn
from ...base import BaseEstimator
import numpy as np
from itertools import product


class DecisionStump(BaseEstimator):
    """
    A decision stump classifier for {-1,1} labels according to the CART algorithm

    Attributes
    ----------
    self.threshold_ : float
        The threshold by which the data is split

    self.j_ : int
        The index of the feature by which to split the data

    self.sign_: int
        The label to predict for samples where the value of the j'th feature is about the threshold
    """

    def __init__(self) -> DecisionStump:
        """
        Instantiate a Decision stump classifier
        """
        super().__init__()
        self.threshold_, self.j_, self.sign_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit a decision stump to the given data. That is, finds the best feature and threshold by which to split

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        best_thr_err = float('inf')
        for feature_idx in range(X.shape[1]):
            (self.threshold_, best_thr_err), self.sign_, self.j_ = \
                min([
                    ((self.threshold_, best_thr_err), self.sign_, self.j_),
                    (self._find_threshold(X[:, feature_idx], y, -1), -1, feature_idx),
                    (self._find_threshold(X[:, feature_idx], y, 1), 1, feature_idx),
                ], key=lambda x: x[0][1])

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict sign responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples

        Notes
        -----
        Feature values strictly below threshold are predicted as `-sign` whereas values which equal
        to or above the threshold are predicted as `sign`
        """
        return ((X[:, self.j_] >= self.threshold_) * 2 - 1) * self.sign_

    def _find_threshold(self, values: np.ndarray, labels: np.ndarray,
                        sign: int) -> Tuple[float, float]:
        """
        Given a feature vector and labels, find a threshold by which to perform a split
        The threshold is found according to the value minimizing the misclassification
        error along this feature

        Parameters
        ----------
        values: ndarray of shape (n_samples,)
            A feature vector to find a splitting threshold for

        labels: ndarray of shape (n_samples,)
            The labels to compare against

        sign: int
            Predicted label assigned to values equal to or above threshold

        Returns
        -------
        thr: float
            Threshold by which to perform split

        thr_err: float between 0 and 1
            Misclassificaiton error of returned threshold

        Notes
        -----
        For every tested threshold, values strictly below threshold are predicted as `-sign` whereas values
        which equal to or above the threshold are predicted as `sign`
        """
        sorted_indexes = np.argsort(values)
        values = values[sorted_indexes]
        labels = labels[sorted_indexes] * sign

        values = np.concatenate(([-np.inf], values, [np.inf]))
        labels = np.concatenate(([0], labels, [0]))

        positive_labels = labels.copy()
        positive_labels[positive_labels < 0] = 0

        negative_labels = labels.copy()
        negative_labels[negative_labels > 0] = 0
        negative_labels *= -1

        cumsum_neg = np.cumsum(negative_labels)
        cumsum_pos = np.cumsum(positive_labels)

        type_one_errors = cumsum_pos - positive_labels
        type_two_errors = cumsum_neg[-1] - cumsum_neg + negative_labels

        errors = (type_one_errors + type_two_errors) / len(sorted_indexes)
        idx_best = np.argmin(errors)
        return values[idx_best], errors[idx_best]

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        from ...metrics import misclassification_error
        return misclassification_error(y, self.predict(X))
