from .templates import make_command, ModelType, CommandType, SubCommand
import os

def test_make_command():
    # chdir to ../..
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    # test commands
    command = make_command(
        command_type=CommandType.EVALUATION.value,
        subcommand=SubCommand.NODE_EFFECT.value,
        case="3",
        model_type=ModelType.InterpBench.value,
        max_len=10,
        categorical_metric="accuracy",
    )
    # assert (
    #     command
    #     == "python main.py eval iit -i 3 --interp-bench --using-wandb --max-len 10 --categorical-metric accuracy"
    # )
    # os.system(command.replace("--using-wandb", ""))
    print(command)


    command = make_command(
        command_type=CommandType.EVALUATION.value,
        subcommand=SubCommand.REALISM.value,
        case="3",
        model_type=ModelType.InterpBench.value,
    )
    # assert command == "python main.py eval realism -i 3 --interp-bench --using-wandb"
    # os.system(command.replace("--using-wandb", ""))
    print(command)
