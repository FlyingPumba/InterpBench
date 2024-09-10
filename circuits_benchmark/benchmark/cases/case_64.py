from typing import Set, Sequence

from circuits_benchmark.benchmark import vocabs
from circuits_benchmark.benchmark.tracr_benchmark_case import TracrBenchmarkCase
from tracr.rasp import rasp


class Case64(TracrBenchmarkCase):
  def get_program(self) -> rasp.SOp:
    return make_cube_each_element()

  def get_task_description(self) -> str:
    return "Cubes each element in the sequence."

  def get_vocab(self) -> Set:
    return vocabs.get_int_numbers_vocab()


def make_cube_each_element() -> rasp.SOp:
  # Apply the cubing function to each element of the input sequence.
  cube_sequence = rasp.Map(lambda x: x ** 3, rasp.tokens).named("cube_sequence")

  return cube_sequence
