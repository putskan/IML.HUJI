import numpy as np
from typing import Tuple
from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    model = AdaBoost(DecisionStump, n_learners).fit(train_X, train_y)
    train_loss = []
    test_loss = []
    t_values = np.arange(1, n_learners + 1)
    for t in t_values:
        train_loss.append(model.partial_loss(train_X, train_y, t))
        test_loss.append(model.partial_loss(test_X, test_y, t))

    go.Figure([
        go.Scatter(x=t_values, y=train_loss, name='train'),
        go.Scatter(x=t_values, y=test_loss, name='test'),
    ],
        layout=dict(xaxis_title='number of learners', yaxis_title='error',
                    title=f'Adaboost Error as a Function of Number of Learners (noise={noise})'),
    ).show()

    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])

    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=([f'T={t}' for t in T]),
                        )

    markers = np.array(['circle', 'x'])
    for i, t in enumerate(T):
        fig.add_trace(decision_surface(lambda X: model.partial_predict(X, t),
                                    xrange=lims[0], yrange=lims[1],
                                    showscale=False),
                      row=i // 2 + 1, col=i % 2 + 1,
                      )

        fig.add_trace(go.Scatter(x=test_X[:, 0], y=test_X[:, 1],
                                 mode='markers',
                                  marker_color='black',
                                  marker_symbol=markers[(test_y > 0).astype(int)],
                                  marker_size=6),
                      row=i // 2 + 1, col=i % 2 + 1,
                      )
    fig.update_layout(
        margin=dict(t=80, b=50),
        title_text=f'Test set & decision boundaries per ensemble size T (noise={noise})',
        height=1000, width=1000,
    )
    fig.update_xaxes(title_text="feature 1")
    fig.update_yaxes(title_text="feature 2")
    fig.show()

    # Question 3: Decision surface of best performing ensemble
    # TODO: redundant. was already calculated
    losses = [model.partial_loss(test_X, test_y, t) for t in range(1, n_learners + 1)]
    best_loss_idx = np.argmin(losses)
    best_t = best_loss_idx + 1
    best_t_acc = np.round(1 - losses[best_loss_idx], 3)

    fig = go.Figure()
    fig.add_trace(decision_surface(lambda X: model.partial_predict(X, best_t),
                                xrange=lims[0], yrange=lims[1],
                                showscale=False),
                  )

    fig.add_trace(go.Scatter(x=test_X[:, 0], y=test_X[:, 1],
                             mode="markers",
                              marker_color='black',
                              marker_symbol=markers[(test_y > 0).astype(int)],
                              marker_size=6),
                  )
    fig.update_layout(
        margin=dict(t=80, b=50),
        title_text=f'Best model - ensemble size: {best_t}, '
                   f'accuracy: {best_t_acc} (noise={noise})'
    )
    fig.update_xaxes(title_text="feature 1")
    fig.update_yaxes(title_text="feature 2")
    fig.show()

    # Question 4: Decision surface with weighted samples
    marker_size = model.D_ / np.max(model.D_) * 25
    fig = go.Figure()
    fig.add_trace(decision_surface(lambda X: model.predict(X),
                                xrange=lims[0], yrange=lims[1],
                                showscale=False),
                  )

    fig.add_trace(go.Scatter(x=train_X[:, 0], y=train_X[:, 1],
                             mode="markers",
                              marker_color='black',
                              marker_symbol=markers[(test_y > 0).astype(int)],
                              marker_size=marker_size),
                  )
    fig.update_layout(
        margin=dict(t=80, b=50),
        title_text=f'Weighted train set (by final D) & model decision boundary (noise={noise})'
    )
    fig.update_xaxes(title_text="feature 1")
    fig.update_yaxes(title_text="feature 2")
    fig.show()

# TODO: color is opposite to them, is it a bug?


if __name__ == '__main__':
    np.random.seed(0)
    for noise in [0., 0.4]:
        fit_and_evaluate_adaboost(noise)
