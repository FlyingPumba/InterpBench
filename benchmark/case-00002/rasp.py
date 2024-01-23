from functools import partial
from typing import Set

import numpy as np
import torch
from datasets import Dataset
from torch import Tensor

from benchmark import vocabs
from benchmark.benchmark_case import BenchmarkCase
from benchmark.common_programs import make_reverse
from benchmark.validation_metrics import l2_metric
from tracr.rasp import rasp
from utils.hooked_tracr_transformer import HookedTracrTransformerBatchInput, HookedTracrTransformer


class Case00002(BenchmarkCase):
  def get_program(self) -> rasp.SOp:
    return make_reverse(rasp.tokens)

  def get_vocab(self) -> Set:
      return vocabs.get_ascii_letters_vocab()

  def get_clean_data(self, count: int = 10) -> Dataset:
    seq_len = self.get_max_seq_len()
    input_data: HookedTracrTransformerBatchInput = []
    output_data: HookedTracrTransformerBatchInput = []

    # set numpy seed
    np.random.seed(self.data_generation_seed)

    vals = list(self.get_vocab())
    for i in range(count):
      permutation = np.random.permutation(vals)
      permutation = permutation[:seq_len - 1]
      input_data.append(["BOS"] + permutation.tolist())
      output_data.append(["BOS"] + permutation[::-1].tolist())

    return self._build_dataset(input_data, output_data)

  def get_validation_metric(self, metric_name: str, tl_model: HookedTracrTransformer) -> Tensor:
    if metric_name not in ["l2"]:
      raise ValueError(f"Metric {metric_name} is not available for case {self}")

    input = self.get_clean_data()[BenchmarkCase.DATASET_INPUT_FIELD]
    with torch.no_grad():
      baseline_output = tl_model(input)

    return partial(l2_metric, baseline_output=baseline_output)
