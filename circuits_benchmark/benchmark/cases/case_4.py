from functools import partial
from typing import Set

import torch
from torch import Tensor

from circuits_benchmark.benchmark import vocabs
from circuits_benchmark.benchmark.benchmark_case import BenchmarkCase
from circuits_benchmark.benchmark.common_programs import make_pair_balance
from circuits_benchmark.metrics.validation_metrics import l2_metric
from tracr.rasp import rasp
from circuits_benchmark.transformers.hooked_tracr_transformer import HookedTracrTransformer


class Case4(BenchmarkCase):
  def get_program(self) -> rasp.SOp:
    return make_pair_balance(rasp.tokens, "(", ")")

  def get_vocab(self) -> Set:
    return vocabs.get_ascii_letters_vocab(count=3).union({"(", ")"})