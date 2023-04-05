from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    underlying_mean = 10
    underlying_var = 1
    sample_size = 1000

    X = np.random.normal(underlying_mean, underlying_var, sample_size)
    ug = UnivariateGaussian()
    ug.fit(X)
    print(f"({ug.mu_}, {ug.var_})")

    # Question 2 - Empirically showing sample mean is consistent
    sizes = np.arange(1, 101) * 10
    expectations = np.empty_like(sizes, dtype=np.float32)

    for i, size in enumerate(sizes):
        ug = UnivariateGaussian()
        ug.fit(X[:size])
        expectations[i] = ug.mu_

    abs_distance = np.abs(expectations - underlying_mean)
    title = "absolute distance between estimates mean and true mean, as function of sample size (showing consistency)"
    figure = px.scatter(x=sizes, y=abs_distance,
                        labels={"x": "sample size",
                                "y": "abs distance between estimated & true mean"},
                        title=title)
    figure.show()

    # Question 3 - Plotting Empirical PDF of fitted model
    ug = UnivariateGaussian()
    ug.fit(X)
    pdfs = ug.pdf(X)
    sorted_indices = pdfs.argsort()
    figure = px.scatter(x=X[sorted_indices], y=pdfs[sorted_indices],
                        labels={"x": "sample value",
                                "y": "estimated pdf value of sample"},
                        title="samples to the estimated gaussian pdf value")
    figure.show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    mu = np.array([0, 0, 4, 0])
    cov_matrix = np.array([
        [1.0, 0.2, 0.0, 0.5],
        [0.2, 2.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.5, 0.0, 0.0, 1.0],
    ])
    samples = np.random.multivariate_normal(mu, cov_matrix, size=1000)
    mv_ge = MultivariateGaussian()
    mv_ge.fit(samples)

    print(mv_ge.mu_)
    print(mv_ge.cov_)

    # Question 5 - Likelihood evaluation
    f_space = np.linspace(-10, 10, 200)
    log_likelihoods = np.empty((len(f_space), len(f_space)))
    for i, f1 in enumerate(f_space):
        for j, f3 in enumerate(f_space):
            log_likelihoods[i, j] = MultivariateGaussian.log_likelihood(
                np.array([f1, 0, f3, 0]), cov_matrix, samples)

    title = "Log-likelihood heatmap, with correspondance to the different f1 & f3 values (different mean)"
    px.imshow(log_likelihoods, title=title,
              labels={"x": "f3", "y": "f1"}).show()

    # Question 6 - Maximum likelihood
    round_to = 3
    i, j = np.unravel_index(np.argmax(log_likelihoods), log_likelihoods.shape)
    f1 = f_space[i].round(round_to)
    f2 = f_space[j].round(round_to)
    print(f"({f1}, {f2})")


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
