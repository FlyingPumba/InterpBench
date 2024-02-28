from typing import Set

from circuits_benchmark.benchmark import vocabs
from circuits_benchmark.benchmark.benchmark_case import BenchmarkCase
from circuits_benchmark.benchmark.common_programs import shift_by
from tracr.rasp import rasp


class Case23(BenchmarkCase):
  def get_program(self) -> rasp.SOp:
    return make_token_pairing(rasp.tokens)

  def get_vocab(self) -> Set:
    return vocabs.get_ascii_letters_vocab(count=3)


def make_token_pairing(sop: rasp.SOp) -> rasp.SOp:
    """
    Pairs adjacent tokens in a sequence.

    Example usage:
      token_pair = make_token_pairing(rasp.tokens)
      token_pair(["a", "b", "c", "d"])
      >> [("a", "b"), ("b", "c"), ("c", "d"), None]
    """
    shifted_sop = shift_by(1, sop)
    token_pair = rasp.SequenceMap(lambda x, y: (x, y) if y is not None else None, sop, shifted_sop)
    return token_pair
