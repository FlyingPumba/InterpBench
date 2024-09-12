from typing import Set

from tracr.rasp import rasp

from circuits_benchmark.benchmark import vocabs
from circuits_benchmark.benchmark.tracr_benchmark_case import TracrBenchmarkCase


class Case33(TracrBenchmarkCase):
  def get_program(self) -> rasp.SOp:
    return make_token_length_parity_checker(rasp.tokens)

  def get_task_description(self) -> str:
    return "Checks if each token's length is odd or even."

  def get_vocab(self) -> Set:
    return vocabs.get_words_vocab()

  def is_trivial(self) -> bool:
      return True


def make_token_length_parity_checker(sop: rasp.SOp) -> rasp.SOp:
    """
    Checks if each token's length is odd or even.

    Example usage:
      length_parity = make_token_length_parity_checker(rasp.tokens)
      length_parity(["hello", "worlds", "!", "2022"])
      >> [False, True, False, True]
    """
    length_parity_checker = rasp.Map(lambda x: len(x) % 2 == 0, sop)
    return length_parity_checker
