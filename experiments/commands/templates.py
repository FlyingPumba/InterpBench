from enum import Enum
from string import Formatter

from .arguments import (
    OptionalArgument,
    OptionalStoreTrueArgument,
    OptionalVariableArgument,
    RequiredArgument,
    VariableArgument,
    StoreTrueArgument,
)


class CommandType(str, Enum):
    CIRCUIT_DISCOVERY = "circuit_discovery"
    EVALUATION = "evaluation"


class ModelType(str, Enum):
  InterpBench = "--interp-bench"
  SIIT_Best = "--siit-weights best --load-from-wandb"
  Natural = "--natural"
  Tracr = "--tracr"

class SubCommand(str, Enum):
  ACDC = "acdc"
  ACDC_LEGACY = "acdc_legacy"
  EAP = "eap"
  INTEGRATED_GRADIENTS = "integrated_gradients"
  NODE_SP = "node_sp"
  EDGE_SP = "edge_sp"
  NODE_EFFECT = "node_effect"
  REALISM = "node_realism"
  GT_CIRCUIT_SCORE = "gt_node_realism"

COMMANDS = {
  "circuit_discovery": {
    "acdc":
    {
        "command":
        """python main.py run acdc -i {case} {model_type} -t {threshold} --abs-value-threshold""",
    },
    "acdc_legacy": {
        "command":
        """python main.py run legacy_acdc -i {case} {model_type} -t {threshold}""",
    },
    "eap": {
        "command":
        """python main.py run eap -i {case} {model_type} -t {threshold}""",
    },
    "integrated_gradients": {
        "command":
        """python main.py run eap -i {case} {model_type} -t {threshold} {steps}""",
        "variable_args": [
          VariableArgument("steps", "--integrated-gradients", 10)
        ]
    },
    "node_sp": {
        "command":
        """python main.py run sp -i {case} {model_type} -t {threshold} {epochs}""",
        "variable_args": [
          VariableArgument("epochs", "--epochs", 3000)
        ]
    },
    "edge_sp": {
        "command":
        """python main.py run sp -i {case} {model_type} -t {threshold} --edgewise --epochs {epochs}""",
        "variable_args": [
          VariableArgument("epochs", "--epochs", 5000)
        ]
    },
    "common_args": [
      RequiredArgument("case"),
      RequiredArgument("model_type"),
      RequiredArgument("threshold"),
      OptionalStoreTrueArgument("same_size", "--same-size"),
      StoreTrueArgument("--using_wandb"),
      StoreTrueArgument("--load-from-wandb")
    ],
    "wandb_info": {
      "project": "circuit_discovery{same_size}",
      "group": """{algorithm}_{case}_{model_suffix}""",
      "name": """{threshold}""",
    }
  },
  "evaluation": {
    "node_effect": {
        "command":
        """python main.py eval iit -i {case} {model_type} {max_len} {categorical_metric}""",
        "variable_args": [
          VariableArgument("max_len", "--max-len", 100),
          VariableArgument("categorical_metric", "--categorical-metric", "accuracy")
        ],
    },
    "node_realism": {
        "command":
        """python main.py eval node_realism -i {case} {model_type}""",
    },
    "gt_node_realism": {
        "command":
        """python main.py eval gt_node_realism -i {case} {model_type} --max-len {max_len}""",
        "variable_args": [
          VariableArgument("max_len", "--max-len", 100)
        ],
        "optional_args": [
          OptionalVariableArgument("relative", "--relative", 1),
        ]
    },
    "common_args": [
      RequiredArgument("case"),
      RequiredArgument("model_type"),
      OptionalStoreTrueArgument("same_size", "--same-size"),
      StoreTrueArgument("--use-wandb")
    ]
  }
}


def make_command(command_type: str, subcommand: str, **kwargs):
  command = COMMANDS[command_type][subcommand]["command"]
  variable_args = COMMANDS[command_type][subcommand].get("variable_args", [])
  common_args = COMMANDS[command_type].get("common_args", [])
  optional_args = COMMANDS[command_type][subcommand].get("optional_args", [])
  all_args = common_args + variable_args + optional_args

  # populate final args and optional args with:
  # 1. default values for non-optional arguments, values from kwargs for required arguments
  # 2. kwargs values for optional arguments if they exist, else do not include them
  final_args = {}
  optional_args = {}
  for arg in all_args:
    if isinstance(arg, OptionalArgument):
      if arg.name in kwargs and isinstance(arg, OptionalVariableArgument):
        value = kwargs.get(arg.name, None)
        optional_args[arg.name] = arg.make(value)
      elif isinstance(arg, OptionalStoreTrueArgument) and arg.name in kwargs and kwargs[arg.name]:
        optional_args[arg.name] = arg.make()
    else:
      if isinstance(arg, RequiredArgument):
        if arg.name not in kwargs:
          raise ValueError(f"Required argument {arg.name} not found in kwargs")
        final_args[arg.name] = arg.make(kwargs[arg.name])
      elif isinstance(arg, StoreTrueArgument):
        optional_args[arg.arg] = arg.make()
      else:
        correct_instance = isinstance(arg, VariableArgument) and not isinstance(arg, OptionalVariableArgument)
        assert correct_instance, RuntimeError(f"Unknown argument type {arg}")
        value = kwargs.get(arg.name, None)
        final_args[arg.name] = arg.make(value)

  # populate the command with the final args
  formatter = Formatter()
  command = formatter.format(command, **final_args)
  for _, value in optional_args.items():
      command += f" {value}"

  return command
