from typing import Set, Sequence

from circuits_benchmark.benchmark import vocabs
from circuits_benchmark.benchmark.tracr_benchmark_case import TracrBenchmarkCase
from tracr.rasp import rasp


class Case41(TracrBenchmarkCase):
  def get_program(self) -> rasp.SOp:
    return make_absolute()

  def get_task_description(self) -> str:
    return "Make each element of the input sequence absolute"

  def get_vocab(self) -> Set:
    return vocabs.get_int_numbers_vocab(min=-10, max=10)


def make_absolute() -> rasp.SOp:
  return rasp.Map(lambda x: x if x >= 0 else -x, rasp.tokens).named("make_absolute")
