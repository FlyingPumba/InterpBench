from typing import Set, Sequence

from circuits_benchmark.benchmark import vocabs
from circuits_benchmark.benchmark.tracr_benchmark_case import TracrBenchmarkCase
from tracr.rasp import rasp


class Case92(TracrBenchmarkCase):
  def get_program(self) -> rasp.SOp:
    return make_zero_if_less_than_previous()

  def get_task_description(self) -> str:
    return "Set each element to 0 if it is less than the previous element."

  def get_vocab(self) -> Set:
    return vocabs.get_int_numbers_vocab()


def make_zero_if_less_than_previous() -> rasp.SOp:
  # Correctly shift the sequence by using Aggregate and Select to create the shifted sequence with the placeholder at the start
  shifted_sequence = rasp.Aggregate(
    rasp.Select(rasp.indices, rasp.indices, lambda k, q: q == k + 1 or k == 0 and q == 0),
    rasp.tokens
  ).named("shifted_sequence_with_placeholder")

  # Use SequenceMap to compare each element with its shifted version
  zero_if_less_than_previous = rasp.SequenceMap(
    lambda original, shifted: 0 if original < shifted else original,
    rasp.tokens,
    shifted_sequence
  ).named("zero_if_less_than_previous")

  return zero_if_less_than_previous
