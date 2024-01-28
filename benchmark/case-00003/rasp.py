from functools import partial
from typing import Set

import torch
import torch.nn.functional as F
from torch import Tensor

from benchmark import vocabs
from benchmark.benchmark_case import BenchmarkCase
from benchmark.common_programs import make_frac_prevs
from benchmark.validation_metrics import kl_divergence, l2_metric
from tracr.rasp import rasp
from utils.hooked_tracr_transformer import HookedTracrTransformer


class Case00003(BenchmarkCase):
  def get_program(self) -> rasp.SOp:
    is_x = (rasp.tokens == "x").named("is_x")
    return make_frac_prevs(is_x)

  def get_vocab(self) -> Set:
    some_letters = vocabs.get_ascii_letters_vocab(count=3)
    some_letters.add("x")
    return some_letters

  def get_max_seq_len(self) -> int:
    return 5

  def get_validation_metric(self, metric_name: str, tl_model: HookedTracrTransformer) -> Tensor:
    if metric_name not in ["l2"]:
      # TODO: Figure out why KL divergence is not working for this case. It seems to be available in ACDC's experiments.
      raise ValueError(f"Metric {metric_name} is not available for case {self}")

    inputs = self.get_clean_data().get_inputs()
    with torch.no_grad():
      baseline_output = tl_model(inputs)
      base_model_logprobs = F.log_softmax(baseline_output, dim=-1)

    if metric_name == "kl":
      return partial(kl_divergence,
                     base_model_logprobs=base_model_logprobs,
                     mask_repeat_candidates=None,
                     last_seq_element_only=False)
    else:
      return partial(l2_metric, baseline_output=baseline_output, is_categorical=False)

