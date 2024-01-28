from functools import partial
from typing import Set

import torch
from torch import Tensor

from benchmark import vocabs
from benchmark.benchmark_case import BenchmarkCase
from benchmark.common_programs import make_pair_balance
from benchmark.validation_metrics import l2_metric
from tracr.rasp import rasp
from utils.hooked_tracr_transformer import HookedTracrTransformer


class Case00004(BenchmarkCase):
  def get_program(self) -> rasp.SOp:
    return make_pair_balance(rasp.tokens, "(", ")")

  def get_vocab(self) -> Set:
    return vocabs.get_ascii_letters_vocab(count=3).union({"(", ")"})

  def get_validation_metric(self, metric_name: str, tl_model: HookedTracrTransformer) -> Tensor:
    if metric_name not in ["l2"]:
      raise ValueError(f"Metric {metric_name} is not available for case {self}")

    inputs = self.get_clean_data().get_inputs()
    with torch.no_grad():
      baseline_output = tl_model(inputs)

    if metric_name == "l2":
      return partial(l2_metric, baseline_output=baseline_output, is_categorical=False)