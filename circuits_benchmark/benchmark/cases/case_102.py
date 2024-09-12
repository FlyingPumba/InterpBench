from typing import Set

from tracr.rasp import rasp

from circuits_benchmark.benchmark import vocabs
from circuits_benchmark.benchmark.tracr_benchmark_case import TracrBenchmarkCase


class Case102(TracrBenchmarkCase):
  def get_program(self) -> rasp.SOp:
    return make_reflect()

  def get_task_description(self) -> str:
    return "Reflects each element within a range (default is [2, 7])."
    # Reflect means that the values will be projected into the range, "bouncing" from the borders, until they have
    # traveled as far in the range as they traveled outside of it."

  def get_vocab(self) -> Set:
    return vocabs.get_int_numbers_vocab(min=-20, max=20)


def reflect_into_range(max, min, x):
  d = max - min
  if x > min and x < max:
    return x
  elif x < min:
    delta = min - x
    i = (delta // d) % 2
    if i == 0:
      return min + (delta % d)
    else:
      return max - (delta % d)
  else:
    delta = x - max
    i = (delta // d) % 2
    if i == 1:
      return min + (delta % d)
    else:
      return max - (delta % d)


def make_reflect(min_val=2, max_val=7) -> rasp.SOp:
  return rasp.Map(lambda x: reflect_into_range(max_val, min_val, x), rasp.tokens)
