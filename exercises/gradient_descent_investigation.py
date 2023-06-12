import numpy as np
import pandas as pd
from typing import Tuple, List, Callable, Type

from IMLearn import BaseModule
from IMLearn.desent_methods import GradientDescent, FixedLR, ExponentialLR
from IMLearn.desent_methods.modules import L1, L2
from IMLearn.learners.classifiers.logistic_regression import LogisticRegression
from IMLearn.utils import split_train_test

import plotly.graph_objects as go
from sklearn import metrics as sklearn_metrics
from IMLearn.model_selection.cross_validate import cross_validate
from IMLearn.metrics import misclassification_error


def plot_descent_path(module: Type[BaseModule],
                      descent_path: np.ndarray,
                      title: str = "",
                      xrange=(-1.5, 1.5),
                      yrange=(-1.5, 1.5)) -> go.Figure:
    """
    Plot the descent path of the gradient descent algorithm

    Parameters:
    -----------
    module: Type[BaseModule]
        Module type for which descent path is plotted

    descent_path: np.ndarray of shape (n_iterations, 2)
        Set of locations if 2D parameter space being the regularization path

    title: str, default=""
        Setting details to add to plot title

    xrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    yrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    Return:
    -------
    fig: go.Figure
        Plotly figure showing module's value in a grid of [xrange]x[yrange] over which regularization path is shown

    Example:
    --------
    fig = plot_descent_path(IMLearn.desent_methods.modules.L1, np.ndarray([[1,1],[0,0]]))
    fig.show()
    """
    def predict_(w):
        return np.array([module(weights=wi).compute_output() for wi in w])

    from utils import decision_surface
    return go.Figure([decision_surface(predict_, xrange=xrange, yrange=yrange, density=70, showscale=False),
                      go.Scatter(x=descent_path[:, 0], y=descent_path[:, 1], mode="markers+lines", marker_color="black")],
                     layout=go.Layout(xaxis=dict(range=xrange),
                                      yaxis=dict(range=yrange),
                                      title=f"GD Descent Path {title}"))


def get_gd_state_recorder_callback() -> Tuple[Callable[[], None], List[np.ndarray], List[np.ndarray]]:
    """
    Callback generator for the GradientDescent class, recording the objective's value and parameters at each iteration

    Return:
    -------
    callback: Callable[[], None]
        Callback function to be passed to the GradientDescent class, recoding the objective's value and parameters
        at each iteration of the algorithm

    values: List[np.ndarray]
        Recorded objective values

    weights: List[np.ndarray]
        Recorded parameters
    """
    # TODO: should i pull from git?
    # TODO: like so?
    values = []
    weights = []

    def callback(**kwargs):
        values.append(kwargs['val'])
        weights.append(kwargs['weights'])

    return callback, values, weights


def compare_fixed_learning_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                 etas: Tuple[float] = (1, .1, .01, .001)):
    # TODO: should be implemented?


    for module in [L1, L2]:
        fig = go.Figure(
            layout=dict(title=f'{module.__name__} value as a function of number of GD iterations',
                        xaxis_title='iteration', yaxis_title='objective function'))

        lowest_loss = np.inf
        lowest_lr = None
        for lr in etas:
            callback, values, weights = get_gd_state_recorder_callback()
            gradient_descent = GradientDescent(FixedLR(lr), callback=callback)
            # TODO: why is this copy needed?
            gradient_descent.fit(module(init.copy()), None, None)  # TODO: None, None?
            plot_descent_path(module, np.array(weights),
                              title=f'lr {lr}, module: {module.__name__}').show()
            fig.add_trace(go.Scatter(x=np.arange(len(values)), y=values,
                                     name=f'{module.__name__} - lr {lr}'))

            curr_min_loss = np.min(values)
            if curr_min_loss < lowest_loss:
                lowest_loss = curr_min_loss
                lowest_lr = lr

        print(f'Module {module.__name__} - lowest loss: {np.round(lowest_loss, 4)}, lowest lr: {lowest_lr}')

        fig.show()

def compare_exponential_decay_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                    eta: float = .1,
                                    gammas: Tuple[float] = (.9, .95, .99, 1)):
    # Optimize the L1 objective using different decay-rate values of the exponentially decaying learning rate
    raise NotImplementedError()

    # Plot algorithm's convergence for the different values of gamma
    raise NotImplementedError()

    # Plot descent path for gamma=0.95
    raise NotImplementedError()


def load_data(path: str = "../datasets/SAheart.data", train_portion: float = .8) -> \
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load South-Africa Heart Disease dataset and randomly split into a train- and test portion

    Parameters:
    -----------
    path: str, default= "../datasets/SAheart.data"
        Path to dataset

    train_portion: float, default=0.8
        Portion of dataset to use as a training set

    Return:
    -------
    train_X : DataFrame of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set

    train_y : Series of shape (ceil(train_proportion * n_samples), )
        Responses of training samples

    test_X : DataFrame of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set

    test_y : Series of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples
    """
    df = pd.read_csv(path)
    df.famhist = (df.famhist == 'Present').astype(int)
    return split_train_test(df.drop(['chd', 'row.names'], axis=1), df.chd, train_portion)


def fit_logistic_regression():
    # Load and split SA Heard Disease dataset
    X_train, y_train, X_test, y_test = load_data()
    # TODO: anything else ^ ?

    # Plotting convergence rate of logistic regression over SA heart disease data
    X_train, y_train, X_test, y_test = X_train.to_numpy(), y_train.to_numpy(), X_test.to_numpy(), y_test.to_numpy()
    logistic_reg = LogisticRegression().fit(X_train, y_train)
    fpr, tpr, thresholds = sklearn_metrics.roc_curve(y_test, logistic_reg.predict_proba(X_test))  # TODO: use x_test right?

    go.Figure(
        data=[go.Scatter(x=fpr, y=tpr, mode='markers+lines', text=thresholds,
                         showlegend=False, marker_size=5)],
        layout=go.Layout(
            title='Logistic Regression ROC Curve',
            xaxis=dict(title=r"$\text{False Positive Rate (FPR)}$"),
            yaxis=dict(title=r"$\text{True Positive Rate (TPR)}$"))).show()  # TODO: clean this function

    optimal_threshold = thresholds[np.argmax(tpr - fpr)]
    logistic_reg.alpha_ = optimal_threshold  # TODO: legit? or re-fit instead
    print(f'No reg: optimal alpha: {np.round(optimal_threshold, 5)}, loss: {np.round(logistic_reg.loss(X_test, y_test), 2)}')


    # Fitting l1- and l2-regularized logistic regression models, using cross-validation to specify values
    # of regularization parameter
    for penalty in ['l1', 'l2']:
        best_loss, best_lam = np.inf, None
        for lam in [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]:
            solver = GradientDescent(learning_rate=FixedLR(1e-4), max_iter=20_000)
            model = LogisticRegression(penalty=penalty, solver=solver, lam=lam)
            _, loss = cross_validate(model, X_train, y_train, scoring=misclassification_error)  # TODO: import is legit right?
            if loss < best_loss:
                best_lam = lam
                best_loss = loss

        loss = LogisticRegression(penalty=penalty, solver=solver, lam=lam).fit(X_train, y_train).loss(X_test, y_test)
        print(f'{penalty}: best lambda {best_lam}, test misclassification error: {np.round(loss, 2)}')


if __name__ == '__main__':
    np.random.seed(0)
    # compare_fixed_learning_rates()  # TODO: uncomment
    # compare_exponential_decay_rates()  # TODO: not needed right?
    fit_logistic_regression()
