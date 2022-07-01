import numpy as np
import scipy.stats as stats


class MultivariateGaussian:
    def __init__(self):
        self.mu_ = None
        self.cov_ = None

    def fit(self, x):
        self.mu_ = x.mean(0)
        self.cov_ = np.matmul((x - self.mu_), (x - self.mu_).T) / x.shape[0]

    def pdf(self, x):
        k = self.mu_.shape[0]  # number of dimensions
        inv_cov = np.linalg.inv(self.cov_)  # inverse of the covariance matrix
        diff = x - self.mu_  # deviation of x from the mean

        term1 = (2*np.pi)**-(k/2)
        term2 = np.linalg.det(self.cov_)**-0.5
        term3 = np.exp(-0.5*np.sum(diff * np.dot(inv_cov, diff), axis=0))

        return term1*term2*term3


    def log_likelihood(self, x):
        k = self.mu_.shape[0]  # number of dimensions
        n = x.shape[1]  # number of samples
        inv_cov = np.linalg.inv(self.cov_)  # inverse of the covariance matrix
        diff = x - self.mu_  # deviation of x from the mean

        term1 = n * k * np.log(2 * np.pi)
        term2 = n * np.log(np.linalg.det(inv_cov))
        term3 = np.sum(diff.T * np.dot(inv_cov, diff.T), axis=0)
        return 0.5 * (term1 + term2 + term3)