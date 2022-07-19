import inspect
import datetime


class OutputGenerator:
    def __init__(self, model, callback):
        self.date = datetime.datetime.today().strftime("%H:%M:%S %d/%m/$Y")
        self.objective_function = self.get_objective(model)
        self.number_of_iterations = callback.shape[0]
        self.results = callback.drop("iteration", axis=1).to_json()

    def get_objective(self, model):
        lines = inspect.getsourcelines(model)
        function = lines[0][-1].replace("return", "").strip()
        return function

    def return_json(self):
        return {
            "meta_data": {
                "date": self.date,
                "objective_function": self.objective_function,
                "number_of_iterations": self.number_of_iterations,
            },
            "results": self.results
        }
