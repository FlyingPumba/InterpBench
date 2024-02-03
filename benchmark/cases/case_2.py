from functools import partial
from typing import Set

import torch
from torch import Tensor

from benchmark import vocabs
from benchmark.benchmark_case import BenchmarkCase
from benchmark.common_programs import make_reverse
from benchmark.validation_metrics import l2_metric
from tracr.rasp import rasp
from utils.hooked_tracr_transformer import HookedTracrTransformer


class Case2(BenchmarkCase):
  def get_program(self) -> rasp.SOp:
    return make_reverse(rasp.tokens)

  def get_vocab(self) -> Set:
      return vocabs.get_ascii_letters_vocab()

  def get_validation_metric(self, metric_name: str, tl_model: HookedTracrTransformer) -> Tensor:
    if metric_name not in ["l2"]:
      raise ValueError(f"Metric {metric_name} is not available for case {self}")

    inputs = self.get_clean_data().get_inputs()
    with torch.no_grad():
      baseline_output = tl_model(inputs)

    return partial(l2_metric, baseline_output=baseline_output)
