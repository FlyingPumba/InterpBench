from functools import partial
from typing import Set

import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset
from torch import Tensor

from benchmark import vocabs
from benchmark.benchmark_case import BenchmarkCase
from benchmark.common_programs import make_frac_prevs
from benchmark.validation_metrics import kl_divergence, l2_metric
from tracr.rasp import rasp
from utils.hooked_tracr_transformer import HookedTracrTransformer, HookedTracrTransformerBatchInput


class Case00003(BenchmarkCase):
  def get_program(self) -> rasp.SOp:
    return make_frac_prevs(rasp.tokens == "x")

  def get_vocab(self) -> Set:
    some_letters = vocabs.get_ascii_letters_vocab(count=3)
    some_letters.add("x")
    return some_letters

  def get_clean_data(self, count: int = 10) -> Dataset:
    seq_len = self.get_max_seq_len()
    input_data: HookedTracrTransformerBatchInput = []
    output_data: HookedTracrTransformerBatchInput = []

    # set numpy seed
    np.random.seed(self.data_generation_seed)

    vals = list(self.get_vocab())
    for _ in range(count):
      sample = np.random.choice(vals, size=seq_len - 1).tolist() # sample with replacement
      output = [sample[:i+1].count("x")/(i+1) for i in range(len(sample))]
      input_data.append(["BOS"] + sample)
      output_data.append(["BOS"] + output)

    return self._build_dataset(input_data, output_data)

  def get_validation_metric(self, metric_name: str, tl_model: HookedTracrTransformer) -> Tensor:
    if metric_name not in ["l2"]:
      # TODO: Figure out why KL divergence is not working for this case. It seems to be available in ACDC's experiments.
      raise ValueError(f"Metric {metric_name} is not available for case {self}")

    input = self.get_clean_data()[BenchmarkCase.DATASET_INPUT_FIELD]
    with torch.no_grad():
      baseline_output = tl_model(input)
      base_model_logprobs = F.log_softmax(baseline_output, dim=-1)

    if metric_name == "kl":
      return partial(kl_divergence,
                     base_model_logprobs=base_model_logprobs,
                     mask_repeat_candidates=None,
                     last_seq_element_only=False)
    else:
      return partial(l2_metric, baseline_output=baseline_output, is_categorical=False)

