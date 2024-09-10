from typing import Set, Sequence

from circuits_benchmark.benchmark import vocabs
from circuits_benchmark.benchmark.tracr_benchmark_case import TracrBenchmarkCase
from tracr.rasp import rasp


class Case66(TracrBenchmarkCase):
  def get_program(self) -> rasp.SOp:
    return make_round()

  def get_task_description(self) -> str:
    return "Round each element in the input sequence to the nearest integer."

  def get_vocab(self) -> Set:
    return vocabs.get_float_numbers_vocab()


def make_round() -> rasp.SOp:
  # Step 1: Add 0.5 to each element, shifting the decimal for rounding.
  shift_for_rounding = rasp.Map(lambda x: x + 0.5, rasp.tokens).named("shift_for_rounding")

  # Step 2: Convert each shifted element to an integer, effectively rounding it.
  rounded_sequence = rasp.Map(lambda x: int(x), shift_for_rounding).named("rounded_sequence")

  return rounded_sequence
