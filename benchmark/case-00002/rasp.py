from functools import partial
from typing import Set

import numpy as np
import torch
from torch import Tensor

from benchmark import vocabs
from benchmark.benchmark_case import BenchmarkCase
from benchmark.case_dataset import CaseDataset
from benchmark.common_programs import make_reverse
from benchmark.validation_metrics import l2_metric
from tracr.rasp import rasp
from utils.hooked_tracr_transformer import HookedTracrTransformerBatchInput, HookedTracrTransformer


class Case00002(BenchmarkCase):
  def get_program(self) -> rasp.SOp:
    return make_reverse(rasp.tokens)

  def get_vocab(self) -> Set:
      return vocabs.get_ascii_letters_vocab()

  def get_clean_data(self, count: int = 10) -> CaseDataset:
    seq_len = self.get_max_seq_len()
    input_data: HookedTracrTransformerBatchInput = []
    output_data: HookedTracrTransformerBatchInput = []

    # set numpy seed
    np.random.seed(self.data_generation_seed)
    vals = sorted(list(self.get_vocab()))

    produce_all = False
    if count is None:
      # produce all possible sequences
      count = len(vals) ** (seq_len - 1)
      produce_all = True

    for index in range(count):
      if produce_all:
        # we want to produce all possible sequences, so we convert the index to base len(vals) and then convert each
        # digit to the corresponding value in vals
        sample = np.base_repr(index, base=len(vals)).zfill(seq_len - 1)
        sample = np.array([vals[int(digit)] for digit in sample])
      else:
        sample = np.random.permutation(vals)
        sample = sample[:seq_len - 1]

      input_data.append(["BOS"] + sample.tolist())
      output_data.append(["BOS"] + sample[::-1].tolist())

    return CaseDataset(input_data, output_data)

  def get_validation_metric(self, metric_name: str, tl_model: HookedTracrTransformer) -> Tensor:
    if metric_name not in ["l2"]:
      raise ValueError(f"Metric {metric_name} is not available for case {self}")

    inputs = self.get_clean_data().get_inputs()
    with torch.no_grad():
      baseline_output = tl_model(inputs)

    return partial(l2_metric, baseline_output=baseline_output)
