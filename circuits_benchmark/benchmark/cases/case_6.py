from typing import Set

from circuits_benchmark.benchmark import vocabs
from circuits_benchmark.benchmark.benchmark_case import BenchmarkCase
from circuits_benchmark.benchmark.common_programs import shift_by
from tracr.rasp import rasp


class Case6(BenchmarkCase):
  def get_program(self) -> rasp.SOp:
    return make_token_oscillation_detector(rasp.tokens)

  def get_task_description(self) -> str:
    return "Detect oscillation patterns in a numeric sequence."

  def get_vocab(self) -> Set:
    return vocabs.get_int_digits_vocab()

  def supports_causal_masking(self) -> bool:
    return False


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
