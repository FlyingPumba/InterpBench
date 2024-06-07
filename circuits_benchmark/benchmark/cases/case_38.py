from typing import Set

from circuits_benchmark.benchmark import vocabs
from circuits_benchmark.benchmark.benchmark_case import BenchmarkCase
from circuits_benchmark.benchmark.common_programs import shift_by
from tracr.rasp import rasp


class Case38(BenchmarkCase):
  def get_program(self) -> rasp.SOp:
    return make_token_alternation_checker(rasp.tokens)

  def get_vocab(self) -> Set:
    return vocabs.get_ascii_letters_vocab(count=3)


def make_token_alternation_checker(sop: rasp.SOp) -> rasp.SOp:
    """
    Checks if tokens alternate between two types.

    Example usage:
      alternation_checker = make_token_alternation_checker(rasp.tokens)
      alternation_checker(["cat", "dog", "cat", "dog"])
      >> [True, True, True, True]
    """
    prev_token = shift_by(1, sop)
    next_token = shift_by(-1, sop)

    prev_token_neq_orig = rasp.SequenceMap(lambda x, y: x != y, prev_token, sop).named("prev_token_neq_orig")
    next_token_neq_orig = rasp.SequenceMap(lambda x, y: x != y, sop, next_token).named("next_token_neq_orig")
    alternation_checker = rasp.SequenceMap(lambda x, y: x and y,
                                           prev_token_neq_orig, next_token_neq_orig).named("alternation_checker")
    return alternation_checker
