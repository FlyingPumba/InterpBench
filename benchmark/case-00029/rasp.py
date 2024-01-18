from typing import Set

from benchmark import vocabs
from benchmark.benchmark_case import BenchmarkCase
from tracr.rasp import rasp


class Case00029(BenchmarkCase):
  def get_program(self) -> rasp.SOp:
    return make_token_abbreviation(rasp.tokens)

  def get_vocab(self) -> Set:
    return vocabs.get_words_vocab()


def make_token_abbreviation(sop: rasp.SOp) -> rasp.SOp:
    """
    Creates abbreviations for each token in the sequence.

    Example usage:
      token_abbreviation = make_token_abbreviation(rasp.tokens)
      token_abbreviation(["international", "business", "machines"])
      >> ["int", "bus", "mac"]
    """
    abbreviation = rasp.Map(lambda x: x[:3] if len(x) > 3 else x, sop)
    return abbreviation
