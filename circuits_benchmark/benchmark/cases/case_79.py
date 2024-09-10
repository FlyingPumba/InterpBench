from typing import Set, Sequence

from circuits_benchmark.benchmark import vocabs
from circuits_benchmark.benchmark.tracr_benchmark_case import TracrBenchmarkCase
from tracr.rasp import rasp


class Case79(TracrBenchmarkCase):
  def get_program(self) -> rasp.SOp:
    return make_check_prime()

  def get_task_description(self) -> str:
    return "Check if each number in a sequence is prime"

  def get_vocab(self) -> Set:
    return vocabs.get_int_numbers_vocab()


def primecheck(n):
  if n < 2:
    return 0
  for i in range(2, int(n ** 0.5) + 1):
    if n % i == 0:
      return 0
  return 1


def make_check_prime() -> rasp.SOp:
  return rasp.Map(lambda x: primecheck(x), rasp.tokens)