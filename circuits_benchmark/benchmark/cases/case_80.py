from typing import Set, Sequence

from circuits_benchmark.benchmark import vocabs
from circuits_benchmark.benchmark.tracr_benchmark_case import TracrBenchmarkCase
from tracr.rasp import rasp


class Case80(TracrBenchmarkCase):
  def get_program(self) -> rasp.SOp:
    return make_element_subtract_constant()

  def get_task_description(self) -> str:
    return "Subtract a constant from each element of the input sequence."

  def get_vocab(self) -> Set:
    return vocabs.get_int_numbers_vocab()


def make_element_subtract_constant(constant=2) -> rasp.SOp:
  subtract_constant = rasp.Map(lambda x: x - constant, rasp.tokens).named(f"subtract_{constant}")

  return subtract_constant
