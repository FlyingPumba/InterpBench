from typing import Set

from circuits_benchmark.benchmark import vocabs
from circuits_benchmark.benchmark.benchmark_case import BenchmarkCase
from tracr.rasp import rasp


class Case35(BenchmarkCase):
  def get_program(self) -> rasp.SOp:
    return make_token_capitalization_alternator(rasp.tokens)

  def get_task_description(self) -> str:
    return "Alternates capitalization of each character in words."

  def get_vocab(self) -> Set:
    return vocabs.get_words_vocab()


def make_token_capitalization_alternator(sop: rasp.SOp) -> rasp.SOp:
    """
    Alternates capitalization of each character in tokens.

    Example usage:
      capitalization_alternator = make_token_capitalization_alternator(rasp.tokens)
      capitalization_alternator(["hello", "world"])
      >> ["HeLlO", "WoRlD"]
    """
    def alternate_capitalization(word):
        return ''.join(c.upper() if i % 2 == 0 else c.lower() for i, c in enumerate(word))

    alternator = rasp.Map(alternate_capitalization, sop)
    return alternator
