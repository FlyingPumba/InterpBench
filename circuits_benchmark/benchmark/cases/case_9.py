from typing import Set

from tracr.rasp import rasp

from circuits_benchmark.benchmark import vocabs
from circuits_benchmark.benchmark.common_programs import make_sort
from circuits_benchmark.benchmark.tracr_benchmark_case import TracrBenchmarkCase


class Case9(TracrBenchmarkCase):
  def get_program(self) -> rasp.SOp:
    return make_sort(rasp.tokens, rasp.tokens, 10, 1)

  def get_task_description(self) -> str:
    return "Sort a list of integers in ascending order."

  def supports_causal_masking(self) -> bool:
    return False

  def get_vocab(self) -> Set:
    return vocabs.get_int_digits_vocab()
