from commands.get_wandb_info import get_wandb_info
from commands.make_command import make_command
from kube_utils import main, get_runs_with_substr
from commands import CommandType, SubCommand, ModelType
from commands.circuit_discovery_config import thresholds, subcommands

def clean_wandb(dry_run: bool):
    print("here")
    wandb_info = get_wandb_info(
        command_type=CommandType.CIRCUIT_DISCOVERY.value,
        subcommand="",
        case="3",
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
    for subcommand in subcommands:
        commands_for_subcommand = []
        for threshold in thresholds:
            command = make_command(
                command_type=command_type,
                subcommand=subcommand,
                model_type=model,
                case="3",
                threshold=threshold,
            )
            commands_for_subcommand.append(command.split(" "))
        all_commands[subcommand] = commands_for_subcommand
    
    commands_to_run = []
    for subcommand, commands in all_commands.items():
        commands_to_run.extend(commands)

    return commands_to_run

if __name__ == "__main__":
    main(build_commands, clean_wandb)