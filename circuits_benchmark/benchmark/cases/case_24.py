from typing import Set

from tracr.rasp import rasp

from circuits_benchmark.benchmark import vocabs
from circuits_benchmark.benchmark.tracr_benchmark_case import TracrBenchmarkCase


class Case24(TracrBenchmarkCase):
  def get_program(self) -> rasp.SOp:
    return make_leading_token_identification(rasp.tokens)

  def get_task_description(self) -> str:
    return "Identifies the first occurrence of each token in a sequence."

  def get_vocab(self) -> Set:
    return vocabs.get_ascii_letters_vocab(count=3)


def make_leading_token_identification(sop: rasp.SOp) -> rasp.SOp:
    """
    Identifies the first occurrence of each token in a sequence.

    Example usage:
      leading_token_id = make_leading_token_identification(rasp.tokens)
      leading_token_id(["x", "y", "x", "z", "y"])
      >> [True, True, False, True, False]
    """
    first_occurrence = rasp.Aggregate(
        rasp.Select(rasp.indices, rasp.indices, rasp.Comparison.EQ),
        sop, default=None).named("first_occurrence")
    return first_occurrence
