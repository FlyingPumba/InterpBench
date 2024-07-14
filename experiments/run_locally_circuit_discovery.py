from argparse import ArgumentParser
import os
from commands import make_command
from commands.circuit_discovery_config import thresholds
from commands import CommandType, ModelType, SubCommand
from commands.get_wandb_info import get_wandb_info
from kube_utils import get_runs_with_substr
from tqdm import tqdm

def clean_wandb(case, 
                algorithm, 
                threshold = None,
                dry_run: bool = False):
    threshold = threshold if threshold is not None else ""
    algorithm = algorithm.value
    wandb_info = get_wandb_info(
        command_type=CommandType.CIRCUIT_DISCOVERY.value,
        subcommand=algorithm,
        case=case,
        model_type=ModelType.InterpBench.value,
        threshold=threshold,
    )
    print(wandb_info)
    runs = get_runs_with_substr(
        project=wandb_info["project"],
        group_substr=wandb_info["group"],
        name_substr=wandb_info["name"],
    )
    for run in runs:
        print(f"Deleting run {run.name} from group {run.group}")
        if not dry_run:
            run.delete()

def run_circuit_discovery_on_interp_bench(
    case: str,
    threshold: float,
    algorithm: SubCommand,
):
    command_type = CommandType.CIRCUIT_DISCOVERY.value
    model = ModelType.InterpBench.value
    
    command = make_command(
        command_type=command_type,
        subcommand=algorithm,
        model_type=model,
        case=case,
        threshold=threshold,
    )
    os.system(command)

def main():
    parser = ArgumentParser()
    parser.add_argument("--case", type=str, required=True)
    parser.add_argument("--threshold", type=float, required=False)
    parser.add_argument("--algorithm", type=SubCommand, required=True)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()
    if args.clean:
        clean_wandb(
            case=args.case,
            algorithm=args.algorithm,
            threshold=args.threshold,
            dry_run=args.dry_run,
        )
    if args.dry_run:
        return

    os.chdir("../")
    if args.threshold is None:
        for threshold in tqdm(thresholds):
            run_circuit_discovery_on_interp_bench(
                case=args.case,
                threshold=threshold,
                algorithm=args.algorithm,
            )
    else:
        run_circuit_discovery_on_interp_bench(
            case=args.case,
            threshold=args.threshold,
            algorithm=args.algorithm,
        )
    

if __name__ == "__main__":
    main()