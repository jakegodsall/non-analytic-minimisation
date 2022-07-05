import numpy as np
import scipy.stats as stats


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
        k = self.mu_.shape[0]  # number of dimensions
        diff = x - self.mu_  # deviation of x from the mean

        term1 = 1. / (np.sqrt(2 * np.pi) ** k * np.linalg.det(self.covmat_))
        term2 = np.exp(-(np.linalg.solve(self.covmat_, diff.T).dot(diff)) / 2)
        return term1 * term2

    def probability_from_mesh(self, xx, yy):
        """
            Generate an array of probability values from a given meshgrid
            of xx and yy
            Note: xx and yy must be the same shape.
        """
        pdf_values = np.zeros((xx.shape[0], yy.shape[0]))

        for i in range(xx.shape[0]):
            for j in range(xx.shape[1]):
                pdf_values[i][j] = mg.pdf([xx[i][j], yy[i][j]])

        return pdf_values

    def log_likelihood(self, x):
        k = self.mu_.shape[0]  # number of dimensions
        n = x.shape[1]  # number of samples
        inv_cov = np.linalg.inv(self.covmat_)  # inverse of the covariance matrix
        diff = x - self.mu_  # deviation of x from the mean
        maha_dist = np.sum(np.tensordot(diff.T, np.dot(inv_cov, diff), 1))

        term1 = n*k*np.log(2*np.pi)
        term2 = n*np.log(np.linalg.det(inv_cov))
        term3 = maha_dist
        return 0.5 * (term1 + term2 + term3)
