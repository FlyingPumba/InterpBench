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

    # set numpy seed
    np.random.seed(self.get_clean_data_seed())

    vals = list(self.get_vocab())
    for i in range(count):
      permutation = np.random.permutation(vals)
      permutation = permutation[:seq_len - 1]
      input_data.append(["BOS"] + permutation.tolist())
      expected_output.append(["BOS"] + permutation[::-1].tolist())

    return input_data, expected_output

  def get_validation_metric(self, tl_model: HookedTracrTransformer) -> str:
    clean_data, _ = self.get_clean_data()
    with torch.no_grad():
      model_out = tl_model(clean_data)
      base_model_logprobs = F.log_softmax(model_out, dim=-1)

    return partial(kl_divergence,
                   base_model_logprobs=base_model_logprobs,
                   mask_repeat_candidates=None,
                   last_seq_element_only=False)