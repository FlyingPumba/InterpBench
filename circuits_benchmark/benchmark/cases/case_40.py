from typing import Set

from tracr.rasp import rasp

from circuits_benchmark.benchmark import vocabs
from circuits_benchmark.benchmark.tracr_benchmark_case import TracrBenchmarkCase


class Case40(TracrBenchmarkCase):
  def get_program(self) -> rasp.SOp:
    return make_sum_digits()

  def get_task_description(self) -> str:
    return "Sum the last and previous to last digits of a number"

  def get_vocab(self) -> Set:
    return vocabs.get_int_numbers_vocab(min=0, max=29)


def make_sum_digits() -> rasp.SOp:
  # Isolate the tens place
  tens_place = rasp.Map(lambda x: x // 10, rasp.tokens).named("tens_place")
  # Isolate the ones place
  ones_place = rasp.Map(lambda x: x % 10, rasp.tokens).named("ones_place")
  # Sum the tens and ones places
  sum_digits = rasp.SequenceMap(lambda x, y: x + y, tens_place, ones_place).named("sum_digits")

  return sum_digits