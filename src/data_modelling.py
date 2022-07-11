import numpy as np


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