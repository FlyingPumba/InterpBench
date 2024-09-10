from typing import Set, Sequence

from circuits_benchmark.benchmark import vocabs
from circuits_benchmark.benchmark.tracr_benchmark_case import TracrBenchmarkCase
from tracr.rasp import rasp


class Case107(TracrBenchmarkCase):
  def get_program(self) -> rasp.SOp:
    return make_wrap()

  def get_task_description(self) -> str:
    return "Wraps each element within a range (make the default range [2, 7])."
    # Wrapping here means that the values are projected into the range starting from the lower bound, once they grow
    # larger than the upper bound, they start again at the lower.

  def get_vocab(self) -> Set:
    return vocabs.get_int_numbers_vocab(min=-15, max=15)


def wrap_into_range(min, max, x):
  # Calculate the size of the range
  range_size = max - min
  # Wrap x into the range
  wrapped_x = ((x - min) % range_size) + min
  return wrapped_x


def make_wrap(min_val=2, max_val=7) -> rasp.SOp:
  return rasp.Map(lambda x: wrap_into_range(min_val, max_val, x), rasp.tokens)