from typing import Set

from benchmark import vocabs
from benchmark.benchmark_case import BenchmarkCase
from benchmark.common_programs import shift_by
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

    alternation_checker = rasp.SequenceMap(lambda x, y: x != y, prev_token, sop)
    alternation_checker = rasp.SequenceMap(lambda x, y: x != y, sop, next_token)
    alternation_checker = rasp.SequenceMap(lambda x, y: x == y, alternation_checker, alternation_checker)

    return alternation_checker
