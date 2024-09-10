from typing import Set, Sequence

from circuits_benchmark.benchmark import vocabs
from circuits_benchmark.benchmark.tracr_benchmark_case import TracrBenchmarkCase
from tracr.rasp import rasp


class Case56(TracrBenchmarkCase):
  def get_program(self) -> rasp.SOp:
    return make_zero_every_third()

  def get_task_description(self) -> str:
    return "Sets every third element to zero."

  def get_vocab(self) -> Set:
    return vocabs.get_int_numbers_vocab()


def make_zero_every_third() -> rasp.SOp:
  # Step 1: Use rasp.indices to generate a sequence of indices.

  # Step 2: Map over the indices to identify every third element, considering 1-based indexing.
  every_third = rasp.Map(lambda x: (x + 1) % 3 == 0, rasp.indices).named("every_third")

  # Step 3: Convert boolean flags (True/False) to 0/1 for easier handling in SequenceMap.
  every_third_numerical = rasp.Map(lambda x: 1 if x else 0, every_third).named("every_third_numerical")

  # Step 4: Use SequenceMap to set every third element to 0 and leave others unchanged.
  result_sequence = rasp.SequenceMap(lambda x, is_third: 0 if is_third else x, rasp.tokens,
                                     every_third_numerical).named("result_sequence")

  return result_sequence
