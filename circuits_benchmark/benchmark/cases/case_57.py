from typing import Set, Sequence

from circuits_benchmark.benchmark import vocabs
from circuits_benchmark.benchmark.tracr_benchmark_case import TracrBenchmarkCase
from tracr.rasp import rasp


class Case57(TracrBenchmarkCase):
  def get_program(self) -> rasp.SOp:
    return make_element_second()

  def get_task_description(self) -> str:
    return "Replaces each element with the second element of the sequence."

  def get_vocab(self) -> Set:
    return vocabs.get_ascii_letters_vocab(count=10)

  def supports_causal_masking(self) -> bool:
    return False


def make_element_second() -> rasp.SOp:
  # Select the second element by matching indices equal to 1.
  second_element_selector = rasp.Select(
    rasp.indices,  # Keys: original indices of the sequence
    rasp.Map(lambda x: 1, rasp.indices),  # Queries: creating a sequence of 1s to match the index 1
    rasp.Comparison.EQ  # Predicate: equality check
  ).named("second_element_selector")

  # Use Aggregate to fill the sequence with the value of the second element.
  second_element_sequence = rasp.Aggregate(
    second_element_selector,  # Selector that identifies the second element
    rasp.tokens,  # SOp: the input sequence
    # Note: default is None as per task rules
  ).named("second_element_sequence")

  return second_element_sequence
