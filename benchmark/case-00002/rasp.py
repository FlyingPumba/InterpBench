from functools import partial
from typing import Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from benchmark import vocabs
from benchmark.benchmark_case import BenchmarkCase
from benchmark.common_programs import make_reverse
from benchmark.validation_metrics import kl_divergence
from tracr.rasp import rasp
from utils.hooked_tracr_transformer import HookedTracrTransformer, HookedTracrTransformerBatchInput


class Case00002(BenchmarkCase):
  def get_program(self) -> rasp.SOp:
    return make_reverse(rasp.tokens)

  def get_vocab(self) -> Set:
      return vocabs.get_ascii_letters_vocab()

  def get_clean_data(self, count: int = 10) -> Tuple[HookedTracrTransformerBatchInput, HookedTracrTransformerBatchInput]:
    seq_len = self.get_max_seq_len()
    input_data: HookedTracrTransformerBatchInput = []
    expected_output: HookedTracrTransformerBatchInput = []

    vals = list(self.get_vocab())
    BOS_id = len(vals)
    for i in range(count):
      permutation = np.random.permutation(vals)
      permutation = permutation[:seq_len]
      input_data.append([BOS_id] + permutation.tolist())
      expected_output.append([BOS_id] + permutation[::-1].tolist())

    return input_data, expected_output