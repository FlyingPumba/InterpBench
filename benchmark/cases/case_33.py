from typing import Set

from benchmark import vocabs
from benchmark.benchmark_case import BenchmarkCase
from tracr.rasp import rasp


class Case33(BenchmarkCase):
  def get_program(self) -> rasp.SOp:
    return make_token_length_parity_checker(rasp.tokens)

  def get_vocab(self) -> Set:
    return vocabs.get_words_vocab()


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
