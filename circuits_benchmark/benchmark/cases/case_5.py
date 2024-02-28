from functools import partial
from typing import Set

import torch
from torch import Tensor

from circuits_benchmark.benchmark.benchmark_case import BenchmarkCase
from circuits_benchmark.benchmark.common_programs import make_shuffle_dyck
from circuits_benchmark.metrics.validation_metrics import l2_metric
from tracr.rasp import rasp
from circuits_benchmark.transformers.hooked_tracr_transformer import HookedTracrTransformer


class Case5(BenchmarkCase):
  def get_program(self) -> rasp.SOp:
    return make_shuffle_dyck(pairs=["()", "{}"])

  def get_vocab(self) -> Set:
    return {"(", ")", "{", "}", "x"}

  def supports_causal_masking(self) -> bool:
    return False

  def get_validation_metric(self, metric_name: str, tl_model: HookedTracrTransformer) -> Tensor:
    if metric_name not in ["l2"]:
      raise ValueError(f"Metric {metric_name} is not available for case {self}")

    inputs = self.get_clean_data().get_inputs()
    with torch.no_grad():
      baseline_output = tl_model(inputs)

    if metric_name == "l2":
      return partial(l2_metric, baseline_output=baseline_output, is_categorical=False)
