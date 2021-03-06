import numpy as np
import pandas as pd


class Simulator:
    def __init__(self, objective):
        self.objective = objective  # define objective function
        self.call_count = 0  # how many times the objective function has been called
        self.callback_count = 0  # number of times callback has been called, also measures iteration count
        self.list_calls_inp = []  # parameter inputs for all calls
        self.list_calls_res = []  # loss value output for all calls
        self.decreasing_list_calls_inp = []  # parameter inputs that resulted in decrease
        self.decreasing_list_calls_res = []  # loss value outputs that resulted in decrease
        self.list_callback_inp = []  # only appends inputs on callback, as such they correspond to the iterations
        self.list_callback_res = []  # only appends results on callback, as such they correspond to the iterations

    def simulate(self, x, *args):
        """
            Executes the simulation.
            Returns the result and updates the lists for the callback.
            Pass to the optimiser as the objective function.
        """
        # evaluate the objective function
        result = self.objective(x, *args)
        # pull the value out of the 1-element numpy array if there is only a single parameter
        x = x[0] if len(x) == 1 else list(x)

        # first call of the simulate function is applied to total list and decreasing list
        if not self.call_count:
            self.decreasing_list_calls_inp.append(x)
            self.decreasing_list_calls_res.append(result)
            self.list_callback_inp.append(x)
            self.list_callback_res.append(result)
        elif result < self.decreasing_list_calls_res[-1]:
            self.decreasing_list_calls_inp.append(x)
            self.decreasing_list_calls_res.append(result)
        self.list_calls_inp.append(x)
        self.list_calls_res.append(result)
        self.call_count += 1
        return result

    def callback(self, xk, *_):
        s1 = ""
        xk = np.atleast_1d(xk)
        for i, x in reversed(list(enumerate(self.list_calls_inp))):
            x = np.atleast_1d(x)
            if np.allclose(x, xk):
                break

        for comp in xk:
            s1 += f"{comp:10.5e}\t"
        s1 += f"{self.list_calls_res[i]:10.5e}"

        self.list_callback_inp.append(xk)
        self.list_callback_res.append(self.list_callss_res[i])

        if not self.callback_count:
            s0 = ""
            for j, _ in enumerate(xk):
                tmp = f"Comp-{j + 1}"
                s0 += f"{tmp:10s}\t"
            s0 += "Objective"
            print(s0)
        print(s1)
        self.callback_count += 1

    def to_dataframe(self):
        data = {}

        params = np.array(self.decreasing_list_calls_inp).T

        if params.ndim == 1:
            data["parameter"] = params
        else:
            for idx, i in enumerate(params):
                data[f"parameter {idx + 1}"] = i
        data["likelihood"] = self.decreasing_list_calls_res

        df = pd.DataFrame(data)
        df["iteration"] = df.index + 1

        return df
