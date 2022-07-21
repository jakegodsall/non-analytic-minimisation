import numpy as np

from scipy.optimize import minimize
from src.callback import Simulator


class UnivariateTestObject:
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
        return rng.normal(mu, variance, size=sample_size)

    def pdf(self, x):
        """
            Return a vector (shape of x) of the probabilities
            of the values of x
        """
        return 1 / (np.sqrt(2 * np.pi * self.variance)) * np.exp(-1 / (2 * self.variance) * (x - self.mu) ** 2)

    def likelihood(self, params, x):
        mu = self.model(params, x)
        likelihood = sum((x - mu) ** 2)
        return likelihood

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


class MultivariateTestObject:
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

    @classmethod
    def change_xdims(self, x):
        return x[:, np.newaxis]

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
        return rng.multivariate_normal(mu, variance, size=sample_size)

    def pdf(self, x):
        """
          Generates the probability of a given x vector based on the
          probability distribution function N(mu_, covmat_)

          Returns: the probability
        """
        x = self.change_xdims(x)  # add a new first dimension to x
        k = self.mu.shape[0]  # number of dimensions
        diff = x - self.mu  # deviation of x from the mean
        inv_covmat = np.linalg.inv(self.variance)

        term1 = (2 * np.pi) ** -(k / 2) * np.linalg.det(inv_covmat)
        term2 = np.exp(-np.einsum('ijk, kl, ijl->ij', diff, inv_covmat, diff) / 2)
        return term1 * term2

    def likelihood(self, params, x):
        mu = params
        covmat = self.variance

        x = self.change_xdims(x)
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