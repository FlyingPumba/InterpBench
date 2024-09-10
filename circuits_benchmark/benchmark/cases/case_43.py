from typing import Set, Sequence

from circuits_benchmark.benchmark import vocabs
from circuits_benchmark.benchmark.tracr_benchmark_case import TracrBenchmarkCase
from tracr.rasp import rasp


class Case43(TracrBenchmarkCase):
  def get_program(self) -> rasp.SOp:
    return make_nth_fibonacci()

  def get_task_description(self) -> str:
    return "Returns the corresponding Fibonacci number for each element in the input sequence."

  def get_vocab(self) -> Set:
    return vocabs.get_int_numbers_vocab(max=20)


def make_nth_fibonacci():
  # Pre-generated Fibonacci sequence up to the 20th number, considering 0th and 1st numbers as 0 and 1.
  fib_sequence = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, 2584, 4181, 6765]

  # Create a sequence of indices for each position in the input sequence
  indices = rasp.Map(lambda x: x, rasp.indices).named("indices")

  # Map each element in the input sequence to its corresponding Fibonacci number
  nth_fib = rasp.Map(lambda x: fib_sequence[x] if x < len(fib_sequence) else 0, rasp.tokens).named("nth_fib")

  return nth_fib