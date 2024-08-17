from circuits_benchmark.commands.build_main_parser import build_main_parser
from circuits_benchmark.commands.train.iit.iit_train import train_model
from circuits_benchmark.utils.get_cases import get_cases
from joblib import Parallel, delayed

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
            delayed(self.update_sweep_results)(hyperparams)
            for hyperparams in self.hyperparams
        )