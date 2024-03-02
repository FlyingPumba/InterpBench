from functools import partial
from typing import Set, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

from circuits_benchmark.benchmark import vocabs
from circuits_benchmark.benchmark.benchmark_case import BenchmarkCase
from circuits_benchmark.benchmark.common_programs import make_frac_prevs
from circuits_benchmark.metrics.validation_metrics import kl_divergence, l2_metric
from tracr.rasp import rasp
from circuits_benchmark.transformers.hooked_tracr_transformer import HookedTracrTransformer


class Case39(BenchmarkCase):
  """Same as Case3 but with increased vocab and max sequence length"""

  def get_program(self) -> rasp.SOp:
    is_x = (rasp.tokens == "x").named("is_x")
    return make_frac_prevs(is_x)

  def get_vocab(self) -> Set:
    some_letters = vocabs.get_ascii_letters_vocab()
    some_letters.add("x")
    return some_letters

  def get_max_seq_len(self) -> int:
    return 60

  def get_correct_output_for_input(self, input: Sequence) -> Sequence:
    """Returns the fraction of 'x' in the input up to the i-th position for all i.
    We define this method so that we don't need to call the original program to get the correct output for each input.
    """
    return [input[:i + 1].count("x") / (i + 1) for i in range(len(input))]
