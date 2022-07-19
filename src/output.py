import inspect
import datetime

import json

from pathlib import Path


class OutputGenerator:
    def __init__(self, model, callback):
        self.date = datetime.datetime.today().strftime("%H:%M:%S %d/%m/%Y")
        self.objective_function = self.get_objective(model)
        self.method = "Nelder-Mead"
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
                "method": self.method,
                "number_of_iterations": self.number_of_iterations,
            },
            "results": self.results
        }


class Saver:
    def __init__(self, save_path):
        self.save_path = Path(".") / save_path

    def save(self, results, file_name):
        self.save_path.mkdir(exist_ok=True)
        with open(self.save_path / file_name, "w", encoding='utf-8') as json_file:
            json.dump(results, json_file, ensure_ascii=False,
                      indent=4)
