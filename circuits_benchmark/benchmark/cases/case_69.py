from typing import Set, Sequence

from circuits_benchmark.benchmark import vocabs
from circuits_benchmark.benchmark.tracr_benchmark_case import TracrBenchmarkCase
from tracr.rasp import rasp


class Case69(TracrBenchmarkCase):
  def get_program(self) -> rasp.SOp:
    return make_sign()

  def get_task_description(self) -> str:
    return "Assign -1, 0, or 1 to each element of the input sequence based on its sign."

  def get_vocab(self) -> Set:
    return vocabs.get_int_numbers_vocab()


def make_sign() -> rasp.SOp:
  # Define the function to apply to each element
  def sign_check(x):
    if x < 0:
      return -1
    elif x > 0:
      return 1
    else:
      return 0

  # Apply the sign checking function to each element of the input sequence
  return rasp.Map(sign_check, rasp.tokens).named("sign")
