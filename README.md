# non-analytic-minimisation

This repository contains all the code for a project involving the comparison
of non-analytic minimisation methods.

The project is structured as follows:

```bash
├── non-analytic-minimisation
│   ├── minimisers # notebooks for experimenting with minimisers
│   ├── README.md
│   ├── results # directory for storing results
│   ├── src
│   │   ├── callback.py # classes for generating a callback and results
│   │   ├── data_modelling.py # classes for test object generation and modelling
│   │   ├── output.py # classes for dealing with results and saving the results
│   ├── univariate_constant_obj.ipynb # minimising univariate gaussian with mu = c
│   └── univariate_non_constant_obj.ipynb # minimising univariate gaussian with m != c

```

The pipeline is as follows.

1. Generate a test object using `data_modelling.TestObject`, from which
a distribution can be sampled using `TestObject.generate_random_sample()`.

2. Define a model for mu.

3. Minimise the -log likelihood `TestObject.likelihood()` using 
`TestObject.minimise()`. 

4. The `TestObject.minimise()` function instantiates a wrapper class 
`callback.Simulator` around the likelihood function to return results for
the entire minimisation process, rather than just the final result.

5. Call `Simulator.to_dataframe()` to return a dataframe of the output of the
minimisation.

6. Instantiate the `output.OutputGenerator` and call 
`OutputGenerator.return_json()` to get a JSON object of all results.

7. Instantiate `output.Saver` and call `Saver.save()` to save the JSON to
disk.