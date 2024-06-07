from typing import Set

from circuits_benchmark.benchmark import vocabs
from circuits_benchmark.benchmark.benchmark_case import BenchmarkCase
from tracr.rasp import rasp


class Case28(BenchmarkCase):
  def get_program(self) -> rasp.SOp:
    return make_token_mirroring(rasp.tokens)

  def get_vocab(self) -> Set:
    return vocabs.get_words_vocab(min_chars=4, max_words=50)


def make_token_mirroring(sop: rasp.SOp) -> rasp.SOp:
    """
    Mirrors each token in the sequence around its central axis.

    Example usage:
      token_mirror = make_token_mirroring(rasp.tokens)
      token_mirror(["abc", "def", "ghi"])
      >> ["cba", "fed", "ihg"]
    """
    mirrored_sop = rasp.Map(lambda x: x[::-1] if x is not None else None, sop)
    return mirrored_sop
