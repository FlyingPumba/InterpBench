from typing import Set

from tracr.rasp import rasp

from circuits_benchmark.benchmark import vocabs
from circuits_benchmark.benchmark.common_programs import shift_by
from circuits_benchmark.benchmark.tracr_benchmark_case import TracrBenchmarkCase


class Case131(TracrBenchmarkCase):
  def get_program(self) -> rasp.SOp:
    return make_pairwise_max()

  def get_task_description(self) -> str:
    return "Makes each element the maximum of it and the previous element, leaving the first element as it is."

  def get_vocab(self) -> Set:
    return vocabs.get_int_numbers_vocab()


def make_pairwise_max() -> rasp.SOp:
  # Shift the input sequence by 1 to the right, filling the first position with fill_value.
  shifted_sequence = shift_by(1, rasp.tokens).named("shifted_sequence")

  # Compare each element of the original sequence with the shifted sequence, taking the maximum.
  pairwise_max = rasp.SequenceMap(lambda x, y: int(max(x, y)), rasp.tokens, shifted_sequence).named("pairwise_max")

  return pairwise_max