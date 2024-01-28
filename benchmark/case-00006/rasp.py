from functools import partial
from typing import Set

import torch
from torch import Tensor

from benchmark import vocabs
from benchmark.benchmark_case import BenchmarkCase
from benchmark.common_programs import shift_by
from benchmark.validation_metrics import l2_metric
from tracr.rasp import rasp
from utils.hooked_tracr_transformer import HookedTracrTransformer


class Case00006(BenchmarkCase):
  def get_program(self) -> rasp.SOp:
    return make_token_oscillation_detector(rasp.tokens)

  def get_vocab(self) -> Set:
    return vocabs.get_int_digits_vocab()

  def get_validation_metric(self, metric_name: str, tl_model: HookedTracrTransformer) -> Tensor:
    if metric_name not in ["l2"]:
      raise ValueError(f"Metric {metric_name} is not available for case {self}")

    inputs = self.get_clean_data().get_inputs()
    with torch.no_grad():
      baseline_output = tl_model(inputs)

    if metric_name == "l2":
      return partial(l2_metric, baseline_output=baseline_output, is_categorical=False)


def make_token_oscillation_detector(sop: rasp.SOp) -> rasp.SOp:
    """
    Detects oscillation patterns in a numeric sequence.

    Example usage:
      oscillation_detector = make_token_oscillation_detector(rasp.tokens)
      oscillation_detector([1, 3, 1, 3, 1])
      >> [True, True, True, True, True]
    """
    prev_token = shift_by(1, sop)
    next_token = shift_by(-1, sop)
    detector_1 = rasp.SequenceMap(lambda x, y: y > x, prev_token, sop)
    detector_2 = rasp.SequenceMap(lambda x, y: y > x, sop, next_token)
    oscillation_detector = rasp.SequenceMap(lambda x, y: x != y, detector_1, detector_2)
    return oscillation_detector
