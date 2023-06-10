from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions
    raise NotImplementedError()

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    raise NotImplementedError()

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    raise NotImplementedError()


def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 1 - Load diabetes dataset and split into training and testing portions
    X, y = datasets.load_diabetes(return_X_y=True)
    train_X, train_y = X[:50], y[:50]
    test_X, test_y = X[50:], y[50:]

    # Question 2 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    lambdas_ridge = np.linspace(0, 0.3, n_evaluations)
    lambdas_lasso = np.linspace(0, 1.1, n_evaluations)
    ridge_loss_train, ridge_loss_val = [], []
    lasso_loss_train, lasso_loss_val = [], []
    for reg_term in lambdas_ridge:
        train_score, val_score = cross_validate(RidgeRegression(reg_term), train_X, train_y, mean_square_error)
        ridge_loss_train.append(train_score)
        ridge_loss_val.append(val_score)

    for reg_term in lambdas_lasso:
        train_score, val_score = cross_validate(Lasso(alpha=reg_term), train_X, train_y, mean_square_error)
        lasso_loss_train.append(train_score)
        lasso_loss_val.append(val_score)

    fig = make_subplots(rows=2, cols=1, subplot_titles=[f'{reg} regression CV mean loss, by regularization term' for reg in ['Ridge', 'Lasso']])
    fig.add_trace(go.Scatter(x=lambdas_ridge, y=ridge_loss_train, name='train (ridge)'), row=1, col=1)
    fig.add_trace(go.Scatter(x=lambdas_ridge, y=ridge_loss_val, name='validation (ridge)'), row=1, col=1)
    fig.add_trace(go.Scatter(x=lambdas_lasso, y=lasso_loss_train, name='train (lasso)'), row=2, col=1)
    fig.add_trace(go.Scatter(x=lambdas_lasso, y=lasso_loss_val, name='validation (lasso)'), row=2, col=1)
    fig.update_xaxes(title_text='lambda')
    fig.update_yaxes(title_text='CV mean loss')
    fig.show()
    # Question 3 - Compare best Ridge model, best Lasso model and Least Squares model
    ridge_best_lambda = lambdas_ridge[np.argmin(ridge_loss_val)]
    lasso_best_lambda = lambdas_lasso[np.argmin(lasso_loss_val)]
    print('ridge best lambda: ', ridge_best_lambda)
    print('lasso best lambda: ', lasso_best_lambda)

    rr_error = RidgeRegression(lam=ridge_best_lambda).fit(train_X, train_y).loss(test_X, test_y)
    lasso_error = mean_square_error(test_y, Lasso(alpha=lasso_best_lambda).fit(train_X, train_y).predict(test_X))
    ols_error = mean_square_error(test_y, LinearRegression().fit(train_X, train_y).predict(test_X))

    print('ridge error: ', rr_error)
    print('lasso error: ', lasso_error)
    print('ols error: ', ols_error)


if __name__ == '__main__':
    np.random.seed(0)
    select_regularization_parameter()
