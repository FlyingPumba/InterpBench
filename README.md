# A benchmark for mechanistic discovery of circuits in Transformers

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.11518575.svg)](https://zenodo.org/doi/10.5281/zenodo.11518575)


## Setup

This project uses [Poetry](https://python-poetry.org/) to manage the dependencies. To install Poetry, you can follow the instructions [here](https://python-poetry.org/docs/#installation).

```bash
git clone --recurse-submodules git@github.com:FlyingPumba/circuits-benchmark.git
poetry env use 3
poetry install
```

Then, to activate the virtual environment: `poetry shell`

## Contents

- `main.py`: Main file to interact with the benchmark.
- `commands/`:  Directory containing the CLI commands that can be used.
- `benchmark/`: Directory containing the cases for the benchmark. Each folder has a `rasp.py` file that contains the RASP code for the case.
- `submodules/`: Directory containing the Git submodules used by the benchmark.
- `tracr/`: A symlink to the `tracr` submodule.
- `acdc/`: A symlink to the `acdc` submodule.

## How to use it

The benchmark is a CLI tool that can be used to run the benchmark on a specific case, or on all the cases. For example, running ACDC on cases with index 1 and 2 can be done with the following command:

```bash
./main.py run acdc -i 1,2 --threshold 0.71
```

The `-i` argument is optional and can be used to specify the cases to run the benchmark on. If not specified, the benchmark will run on all the cases.
To check the arguments available for a specific command, you can use the `--help` flag. For example, for ACDC:

```bash
./main.py run acdc --help
```

After running an algorith, the output can be found in the `results` folder.

## Compilation

The benchmark CLI also provides a `compile` commmand that can be used to preemtively compile the RASP code for all the cases into their corresponding Tracr/TransformerLends models. This can be useful to speed up the benchmark, as the compilation can take a long time. The compilation can be done with the following command:

```bash
./main.py compile
```

## Docker image

To build the Docker image:

```bash
docker build . -t circuits-benchmark
```

And to run it:

```bash
docker run circuits-benchmark <CLI arguments>
```

## Tests

To run the tests, you can just run `pytest` in the root directory of the project. The tests for submodules are ignored by default (see `pytest.ini` file).
If you want to run specific tests, you can use the `-k` flag: `pytest -k "get_cases_test"`.
