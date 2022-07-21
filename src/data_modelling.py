import numpy as np

from scipy.optimize import minimize
from src.callback import Simulator


class TestObject:
    def __init__(self):
        # random number generation hyperparameters
        self.random_state = None
        self.mu = None
        self.variance = None
        self.sample_size = None
        # optimisation hyperparameters
        self.model = None
        self.initial_guess = None
        self.method = None
        self.tolerance = None

    def generate_random_sample(self,
                               random_state,
                               mu,
                               variance,
                               sample_size):
        # set object variables
        self.random_state = random_state
        self.mu = mu
        self.variance = variance
        self.sample_size = sample_size

        # set the random seed
        rng = np.random.default_rng(random_state)

        # generate a random sample using the numpy function
        sample = rng.normal(mu, variance, sample_size)

        return sample

    # define the uni-variate Gaussian likelihood function
    def univariate_likelihood(self, params, x):
        mu = self.model(params, x)
        likelihood = sum((x - mu) ** 2)
        return likelihood

    def multivariate_likelihood(self, params, x):
        mu = params
        covmat = self.variance

        x = x[:, np.newaxis]
        k = mu.shape[0]
        n = x.shape[1]
        inv_covmat = np.linalg.inv(covmat)
        diff = x - mu
        maha_dist = np.einsum('ijk, kl, ijl->ij', diff, inv_covmat, diff)
        return maha_dist



    def minimise(self, x, initial_guess, method, tolerance=1e-6):
        self.initial_guess = initial_guess
        self.method = method
        self.tolerance = tolerance

        # instantiate the simulator class
        lik_sim = Simulator(self.likelihood)

        # minimise the -log(L) function using the wrapper class Simulator
        lik_model = minimize(lik_sim.simulate,
                             x0=initial_guess,
                             args=(x),
                             method=method,
                             tol=tolerance,
                             options={"disp": True})

        return lik_model


class MultivariateGaussian:
    """
        Class for fitting a multivariate gaussian distribution to
        a nxd array where n = the number of samples and d is the number of
        random variables or parameters.
    """

    def __init__(self):
        self.mu_ = None
        self.covmat_ = None

    def fit(self, x):
        """
            Fits the mean and covariance matrix for given input data x
            to attributes mu_ and covmat_ respectively.
        """
        self.mu_ = x.mean(0)
        self.covmat_ = np.dot((x - self.mu_).T, (x - self.mu_)) / x.shape[0]

    def pdf(self, x):
        """
          Generates the probability of a given x vector based on the
          probability distribution function N(mu_, covmat_)

          Returns: the probability
        """
        x = x[:, np.newaxis]  # add a new first dimension to x
        k = self.mu_.shape[0]  # number of dimensions
        diff = x - self.mu_  # deviation of x from the mean
        inv_covmat = np.linalg.inv(self.covmat_)

        term1 = (2 * np.pi) ** -(k / 2) * np.linalg.det(inv_covmat)
        term2 = np.exp(-np.einsum('ijk, kl, ijl->ij', diff, inv_covmat, diff) / 2)
        return term1 * term2