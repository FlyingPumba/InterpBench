from typing import Set, Sequence

from circuits_benchmark.benchmark import vocabs
from circuits_benchmark.benchmark.common_programs import shift_by
from circuits_benchmark.benchmark.tracr_benchmark_case import TracrBenchmarkCase
from tracr.rasp import rasp


class Case118(TracrBenchmarkCase):
  def get_program(self) -> rasp.SOp:
    return make_pairwise_sum()

  def get_task_description(self) -> str:
    return "Replaces each element with the sum of it and the previous element."

  def get_vocab(self) -> Set:
    return vocabs.get_int_numbers_vocab()

  def supports_causal_masking(self) -> bool:
    return False


def make_pairwise_sum() -> rasp.SOp:
  # Shift the input sequence by 1 to the right, filling the first position with fill_value.
  shifted_sequence = rasp.SequenceMap(lambda x, y: x if y > 0 else 0,
                                      shift_by(1, rasp.tokens).named("shifted_sequence"), rasp.indices)

  # Compare each element of the original sequence with the shifted sequence, taking the maximum.
  pairwise_sum = rasp.SequenceMap(lambda x, y: int(x + y), rasp.tokens, shifted_sequence).named("pairwise_max")

  return pairwise_sum
