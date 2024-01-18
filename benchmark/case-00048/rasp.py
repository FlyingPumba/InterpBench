from typing import Set

from benchmark import vocabs
from benchmark.benchmark_case import BenchmarkCase
from benchmark.common_programs import shift_by
from tracr.rasp import rasp


class Case00048(BenchmarkCase):
  def get_program(self) -> rasp.SOp:
    return make_token_oscillation_detector(rasp.tokens)

  def get_vocab(self) -> Set:
    return vocabs.get_int_digits_vocab()


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
    oscillation_detector = rasp.SequenceMap(lambda x, y: y > x, prev_token, sop)
    oscillation_detector = rasp.SequenceMap(lambda x, y: y > x, sop, next_token)
    oscillation_detector = rasp.SequenceMap(lambda x, y: x != y, oscillation_detector, oscillation_detector)
    return oscillation_detector
