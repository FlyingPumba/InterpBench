from typing import Set

from benchmark import vocabs
from benchmark.benchmark_case import BenchmarkCase
from tracr.rasp import rasp


class Case26(BenchmarkCase):
  def get_program(self) -> rasp.SOp:
    return make_token_cascade(rasp.tokens)

  def get_vocab(self) -> Set:
    return vocabs.get_ascii_letters_vocab(count=3)


def make_token_cascade(sop: rasp.SOp) -> rasp.SOp:
    """
    Creates a cascading effect by repeating each token in sequence incrementally.

    Example usage:
      token_cascade = make_token_cascade(rasp.tokens)
      token_cascade(["a", "b", "c"])
      >> ["a", "bb", "ccc"]
    """
    cascade_sop = rasp.SequenceMap(lambda x, i: x * (i + 1), sop, rasp.indices)
    return cascade_sop
