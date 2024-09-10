from typing import Set, Sequence

from circuits_benchmark.benchmark import vocabs
from circuits_benchmark.benchmark.tracr_benchmark_case import TracrBenchmarkCase
from tracr.rasp import rasp


class Case72(TracrBenchmarkCase):
  def get_program(self) -> rasp.SOp:
    return make_negation()

  def get_task_description(self) -> str:
    return "Negate each element in the input sequence."

  def get_vocab(self) -> Set:
    return vocabs.get_int_numbers_vocab(min=-10, max=10)


def make_negation() -> rasp.SOp:
  # Define the negation function
  negation_function = lambda x: -x

  # Apply the negation function element-wise to the input sequence
  negated_sequence = rasp.Map(negation_function, rasp.tokens).named("negated_sequence")

  return negated_sequence
