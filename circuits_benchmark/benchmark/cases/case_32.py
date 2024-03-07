from typing import Set

from circuits_benchmark.benchmark import vocabs
from circuits_benchmark.benchmark.benchmark_case import BenchmarkCase
from circuits_benchmark.benchmark.common_programs import shift_by
from tracr.rasp import rasp


class Case32(BenchmarkCase):
  def get_program(self) -> rasp.SOp:
    return make_token_boundary_detector(rasp.tokens)

  def get_vocab(self) -> Set:
    return vocabs.get_words_vocab()


def make_token_boundary_detector(sop: rasp.SOp) -> rasp.SOp:
    """
    Detects the boundaries between different types of tokens in a sequence.

    Example usage:
      token_boundary = make_token_boundary_detector(rasp.tokens)
      token_boundary(["apple", "banana", "apple", "orange"])
      >> [False, True, False, True]
    """
    previous_token = shift_by(1, sop)
    boundary_detector = rasp.SequenceMap(
        lambda x, y: x != y, sop, previous_token)
    return boundary_detector
