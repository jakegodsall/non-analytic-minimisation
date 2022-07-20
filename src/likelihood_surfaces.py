import numpy as np
import matplotlib.pyplot as plt
import inspect


class Model:
    def __init__(self, model_fn):
        self.model_fn = model_fn
        self.model_name = self.get_model_name(model_fn)

    def generate_random_sample(self,
                               random_state,
                               mu,
                               variance,
                               sample_size):

        # set the random seed
        rng = np.random.default_rng(random_state)

        # generate a random sample using the numpy function
        sample = rng.normal(mu, variance, sample_size)

        return sample

    def get_model_name(self, model):
        lines = inspect.getsourcelines(model)
        function = lines[0][-1].replace("return", "").strip()
        return function

    def likelihood(self, params, x=None):
        mu = self.model_fn(params, x)
        sigma = 2
        n = len(x)
        L = (n / 2) * np.log(2 * np.pi) + (n / 2) * np.log(sigma ** 2) + (1 / (2 * sigma ** 2)) * sum((x - mu) ** 2)
        return L

    def plot(self, x, xlims=(-10, 10), ylims=(-10, 10), step=.1):

        fig = plt.Figure(figsize=(20, 10))

        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

        xx, yy = np.meshgrid(np.arange(*xlims, step), np.arange(*ylims, step))

        values = np.array((xx, yy)).T.reshape(-1, 2)

        # FUNCTION PLOT

        function_values = np.zeros(values.shape[0])

        for idx, val in enumerate(values):
            function_values[idx] = self.model_fn(val, x)

        contours = ax1.contour(xx, yy, function_values.reshape(xx.shape), 10)
        ax1.clabel(contours, inline=1, fontsize=12)
        ax1.set_ylabel("y", fontsize=25)
        ax1.set_xlabel("x", fontsize=25)
        ax1.set_title(self.model_name, fontsize=25)

        # LIKELIHOOD PLOT

        likelihood_values = np.zeros(values.shape[0])

        for idx, val in enumerate(values):
            likelihood_values[idx] = self.likelihood(val, x)

        contours = ax2.contour(xx, yy, likelihood_values.reshape(xx.shape), 10)
        ax2.clabel(contours, inline=1, fontsize=12)
        ax2.set_ylabel("y", fontsize=25)
        ax2.set_xlabel("x", fontsize=25)
        ax2.set_title("Likelihood", fontsize=25)

        return fig