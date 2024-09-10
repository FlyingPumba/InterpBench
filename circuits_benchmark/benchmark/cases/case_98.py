from typing import Set, Sequence

from circuits_benchmark.benchmark import vocabs
from circuits_benchmark.benchmark.common_programs import make_length
from circuits_benchmark.benchmark.tracr_benchmark_case import TracrBenchmarkCase
from tracr.rasp import rasp


class Case98(TracrBenchmarkCase):
  def get_program(self) -> rasp.SOp:
    return make_max_element()

  def get_task_description(self) -> str:
    return "Return a sequence with the maximum element repeated."

  def get_vocab(self) -> Set:
    return vocabs.get_int_numbers_vocab()

  def supports_causal_masking(self) -> bool:
    return False


def make_max_element() -> rasp.SOp:
  # A selector comparing each element with every other element using LEQ (less than or equal)
  unique_tokens = rasp.SequenceMap(lambda x, y: x + y * 0.00000000000001, rasp.tokens, rasp.indices)
  leq_selector = rasp.Select(unique_tokens, unique_tokens, rasp.Comparison.LEQ).named("leq_selector")
  # Counting the number of elements each element is less than or equal to
  leq_count = rasp.SelectorWidth(leq_selector).named("leq_count")
  # The maximum element is the one that is less or equal to all elements (count equal to sequence length)
  length_sop = make_length()
  max_element_selector = rasp.Select(leq_count, length_sop, rasp.Comparison.EQ).named("max_element_selector")
  # Using Aggregate to select the maximum element and broadcast it across the entire sequence
  max_sequence = rasp.Aggregate(max_element_selector, rasp.tokens).named("max_sequence")
  return max_sequence
