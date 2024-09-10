from typing import Set, Sequence

from circuits_benchmark.benchmark import vocabs
from circuits_benchmark.benchmark.tracr_benchmark_case import TracrBenchmarkCase
from tracr.rasp import rasp


class Case61(TracrBenchmarkCase):
  def get_program(self) -> rasp.SOp:
    return make_rank()

  def get_task_description(self) -> str:
    return "Ranks each element according to its size."

  def get_vocab(self) -> Set:
    return vocabs.get_int_numbers_vocab()

  def supports_causal_masking(self) -> bool:
    return False


def make_rank() -> rasp.SOp:
  # Selector that creates a comparison matrix where each element is compared to every other element to find how many are smaller.
  less_than_selector = rasp.Select(rasp.tokens, rasp.tokens, rasp.Comparison.LT).named("less_than_selector")

  # Count the number of elements that are smaller than each element.
  smaller_count = rasp.SelectorWidth(less_than_selector).named("smaller_count")

  # Since ranks start from 1, add 1 to each count to get the rank.
  rank = rasp.Map(lambda x: x + 1, smaller_count).named("rank")

  return rank
