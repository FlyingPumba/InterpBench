from typing import Set, Sequence

from circuits_benchmark.benchmark import vocabs
from circuits_benchmark.benchmark.tracr_benchmark_case import TracrBenchmarkCase
from tracr.rasp import rasp


class Case62(TracrBenchmarkCase):
  def get_program(self) -> rasp.SOp:
    return make_factorial()

  def get_task_description(self) -> str:
    return "Replaces each element with its factorial."

  def get_vocab(self) -> Set:
    return vocabs.get_int_numbers_vocab()


def make_factorial() -> rasp.SOp:
  factorial = rasp.Map(lambda x: factorial_helper(x), rasp.tokens).named("factorial")
  return factorial


def factorial_helper(n: int) -> int:
  # Placeholder for factorial calculation
  # In actual RASP code, this function cannot exist due to RASP's limitations.
  # This represents a conceptual step that needs a workaround.
  if n == 0:
    return 1
  else:
    return n * factorial_helper(n - 1)