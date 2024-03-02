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
