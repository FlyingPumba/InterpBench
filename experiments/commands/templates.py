from enum import Enum

from .arguments import (
    OptionalStoreTrueArgument,
    OptionalVariableArgument,
    RequiredArgument,
    VariableArgument,
    StoreTrueArgument,
)


class CommandType(str, Enum):
    CIRCUIT_DISCOVERY = "circuit_discovery"
    EVALUATION = "evaluation"


def get_model_suffix(model_type: str, case: str) -> str:
    if model_type == ModelType.InterpBench.value:
        return "interp_bench"
    elif model_type == ModelType.SIIT_Best.value:
        from circuits_benchmark.utils.ll_model_loader.best_weights import get_best_weight
        return f"siit_{get_best_weight(case)}"
    elif model_type == ModelType.Natural.value:
        return "natural"
    elif model_type == ModelType.Tracr.value:
        return "tracr"
    else:
        raise ValueError(f"Unknown model type {model_type}")

class ModelType(str, Enum):
  InterpBench = "--interp-bench"
  SIIT_Best = "--siit-weights best --load-from-wandb"
  Natural = "--natural"
  Tracr = "--tracr"

class SubCommand(str, Enum):
  ACDC = "acdc"
  ACDC_LEGACY = "acdc_legacy"
  EAP = "eap"
  INTEGRATED_GRADIENTS = "integrated_grad"
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
        """python main.py run eap -i {case} {model_type} --threshold {threshold} --abs-val-threshold {regression_loss} {classification_loss}""",
        "variable_args": [
          VariableArgument("regression_loss", "--regression-loss", "mae"),
          VariableArgument("classification_loss", "--classification-loss", "kl_div")
        ]
    },
    "integrated_grad": {
        "command":
        """python main.py run eap -i {case} {model_type} --threshold {threshold} --abs-val-threshold {regression_loss} {classification_loss} {steps}""",
        "variable_args": [
          VariableArgument("steps", "--integrated-grad-steps", 10),
          VariableArgument("regression_loss", "--regression-loss", "mae"),
          VariableArgument("classification_loss", "--classification-loss", "kl_div"),
        ]
    },
    "node_sp": {
        "command":
        """python main.py run sp -i {case} {model_type} --lambda-reg {threshold} {epochs}""",
        "variable_args": [
          VariableArgument("epochs", "--epochs", 5000)
        ]
    },
    "edge_sp": {
        "command":
        """python main.py run sp -i {case} {model_type} --lambda-reg {threshold} --edgewise {epochs}""",
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
