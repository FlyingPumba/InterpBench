from typing import Set

from tracr.rasp import rasp

from circuits_benchmark.benchmark import vocabs
from circuits_benchmark.benchmark.tracr_benchmark_case import TracrBenchmarkCase


class Case42(TracrBenchmarkCase):
  def get_program(self) -> rasp.SOp:
    return make_first_element()

  def get_task_description(self) -> str:
    return "Return a sequence composed only of the first element of the input sequence."

  def get_vocab(self) -> Set:
    return vocabs.get_ascii_letters_vocab(count=10)


def make_first_element() -> rasp.SOp:
  # Selector that identifies the first element by comparing indices to 0.
  first_elem_selector = rasp.Select(rasp.indices, rasp.Map(lambda x: 0, rasp.indices), rasp.Comparison.EQ).named(
    "first_elem_selector")

  # Use Aggregate to broadcast the first element across the entire sequence.
  first_element_sequence = rasp.Aggregate(first_elem_selector, rasp.tokens, default=None).named(
    "first_element_sequence")

  return first_element_sequence