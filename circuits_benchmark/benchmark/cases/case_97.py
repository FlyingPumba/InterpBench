from typing import Set

from tracr.rasp import rasp

from circuits_benchmark.benchmark import vocabs
from circuits_benchmark.benchmark.cases.case_98 import make_max_element
from circuits_benchmark.benchmark.tracr_benchmark_case import TracrBenchmarkCase


class Case97(TracrBenchmarkCase):
  def get_program(self) -> rasp.SOp:
    return make_scale_by_max()

  def get_task_description(self) -> str:
    return "Scale a sequence by its maximum element."

  def get_vocab(self) -> Set:
    return vocabs.get_int_numbers_vocab()

  def supports_causal_masking(self) -> bool:
    return False


def make_scale_by_max() -> rasp.SOp:
  # Find the maximum element in the sequence.
  max_element = make_max_element().named("max_element")

  # Assume the maximum element is not zero to avoid division by zero.
  # Divide each element in the sequence by the maximum element.
  scale_by_max_sequence = rasp.SequenceMap(lambda x, y: (x / y) if y > 0 else 0, rasp.tokens, max_element).named(
    "scale_by_max_sequence")

  return scale_by_max_sequence
