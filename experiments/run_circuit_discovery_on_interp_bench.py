from commands.get_wandb_info import get_wandb_info
from commands.make_command import make_command
from kube_utils import main, get_runs_with_substr
from commands import CommandType, ModelType
from commands.circuit_discovery_config import thresholds, subcommands
from circuits_benchmark.utils.get_cases import get_names_of_working_cases


working_cases = sorted(get_names_of_working_cases())
print(working_cases)

ioi = [case for case in working_cases if "ioi" in case]
for ioi_case in ioi:
    working_cases.remove(ioi_case)

def clean_wandb(dry_run: bool):
    for case in working_cases:
        wandb_info = get_wandb_info(
            command_type=CommandType.CIRCUIT_DISCOVERY.value,
            subcommand="",
            case=case,
            model_type=ModelType.InterpBench.value,
            threshold="",
        )
        print(wandb_info)
        runs = get_runs_with_substr(
            project=wandb_info["project"],
            group_substr=wandb_info["group"],
            name_substr="",
        )
        for run in runs:
            print(f"Deleting run {run.name} from group {run.group}")
            if not dry_run:
                run.delete()


def build_commands() -> list[list[str]]:
    command_type = CommandType.CIRCUIT_DISCOVERY.value
    model = ModelType.InterpBench.value
    all_commands = {}
    for case in working_cases:
        commands_for_case = []
        for subcommand in subcommands:
            for threshold in thresholds:
                command = make_command(
                    command_type=command_type,
                    subcommand=subcommand,
                    model_type=model,
                    case=case,
                    threshold=threshold,
                )
                commands_for_case.append(command.split(" "))
        all_commands[case] = commands_for_case
    
    commands_to_run = []
    for _, commands in all_commands.items():
        commands_to_run.extend(commands)
    # dump commands to log file
    with open("commands_to_run.log", "w") as f:
        for command in commands_to_run:
            f.write(" ".join(command) + "\n")
    return commands_to_run

if __name__ == "__main__":
    main(build_commands, clean_wandb)