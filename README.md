# InterpBench: Semi-Synthetic Transformers for Evaluating Mechanistic Interpretability Techniques

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.11518575.svg)](https://zenodo.org/doi/10.5281/zenodo.11518575)

This Python project provides a framework for creating and evaluating the models in InterpBench, a collection of semi-synthetic transformers with known circuits for evaluating mechanistic interpretability techniques

## Setup

This project can be setup by either downloading it and installing the dependencies, or by using the [Docker image](docker.io/iarcuschin/circuits-benchmark:latest). 
We use [Poetry](https://python-poetry.org/) to manage the dependencies, which you can install by following the instructions [here](https://python-poetry.org/docs/#installation).

Run the following Bash commands to download the project and its dependencies:
```bash
git clone --recurse-submodules git@github.com:FlyingPumba/circuits-benchmark.git
cd circuits-benchmark
poetry env use 3
poetry install
```

Then, to activate the virtual environment: `poetry shell`

## Usage

You can either use InterpBench by downloading the pre-trained models from [the Hugging Face repository](https://huggingface.co/cybershiptrooper/InterpBench) (see an example [here](https://colab.research.google.com)), or by running the commands available in the Python framework.

See [EXPERIMENTS.md](EXPERIMENTS.md) for a list of the commands used in the paper's empirical study.

## Tests

To run the tests, you can just run `pytest` in the root directory of the project. The tests for submodules are ignored by default.
If you want to run specific tests, you can use the `-k` flag: `pytest -k "get_cases_test"`.
