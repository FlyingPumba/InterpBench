import os
import pickle
from argparse import ArgumentParser, Namespace

from joblib import Parallel, delayed

import wandb
from circuits_benchmark.commands.build_main_parser import build_main_parser
from circuits_benchmark.commands.train.iit.iit_train import train_model
from circuits_benchmark.utils.get_cases import get_cases


class Sweep:
    def __init__(
        self,
        task_idx: str,
        type_of_hyperparam: str,
        list_of_values: list[str],
        control_params: list[str] = [],
    ):
        self.task_idx = task_idx
        self.hyperparams = self.make_list_of_hyperparams(
            type_of_hyperparam, list_of_values, control_params
        )
        print("Hyperparameters to sweep: ", self.hyperparams)
        self.sweep_results = None

    @staticmethod
    def make_list_of_hyperparams(
        type_of_hyperparam: str,
        list_of_values: list[str],
        control_params: list[str] = [],
    ):
        return [
            [f"--{type_of_hyperparam}", str(value)] + control_params for value in list_of_values
        ]

    def update_sweep_results(self, hyperparams: list[str]):
        task_idx = self.task_idx
        task = get_cases(indices=task_idx)[0]
        args = build_main_parser().parse_known_args(
            ["train", "iit", "--task", task_idx] + hyperparams
        )[0]
        args.atol = 0.05
        args.lr_scheduler = ""
        args.detach_while_caching = not args.backprop_on_cache

        model_pair = train_model(task, args)
        metrics = {
            "train metrics": model_pair.train_metrics,
            "val metrics": model_pair.test_metrics,
            "stop epoch": model_pair.stopping_epoch,
            "args": args,
        }
        return metrics

    def run_sweep(self, n_processes: int = 4):
        self.sweep_results = Parallel(n_jobs=n_processes)(
            delayed(self.update_sweep_results)(hyperparams) for hyperparams in self.hyperparams
        )


def get_sweep_results_for_seeds(task_idx, seeds, strict_weight, epochs = 20):
    seeds_sweep_expt = Sweep(
        task_idx,
        "seed",
        seeds,
        control_params=["--epochs", f"{epochs}", "--strict_weight", str(strict_weight)],
    )
    seeds_sweep_expt.run_sweep(5)
    seed_results = seeds_sweep_expt.sweep_results
    sweep_results = {seeds[i]: seed_results[i] for i in range(len(seeds))}
    return sweep_results


def main(args: Namespace):
    task = args.task
    seeds = range(args.num_seeds)
    strict_weight = args.strict_weight
    sweep_results = get_sweep_results_for_seeds(task, seeds, strict_weight, epochs=args.epochs)

    save_dir = f"results/sweeps/{task}/{strict_weight}"
    os.makedirs(save_dir, exist_ok=True)

    save_file = f"{save_dir}/sweep_results.pkl"
    with open(save_file, "wb") as f:
        pickle.dump(sweep_results, f)

    if args.use_wandb:
        wandb.init(
            project="SIIT_sweep", 
            name=f"{task}_{strict_weight}",
            group=f"{task}",
        )
        wandb.save(f"{save_file}", base_path=".")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-i", "--task", type=str, required=True)
    parser.add_argument("-s", "--strict_weight", type=float, required=True)
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--num-seeds", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()
    main(args)
