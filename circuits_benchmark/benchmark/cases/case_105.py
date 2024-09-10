from typing import Set, Sequence

from circuits_benchmark.benchmark import vocabs
from circuits_benchmark.benchmark.tracr_benchmark_case import TracrBenchmarkCase
from tracr.rasp import rasp


class Case105(TracrBenchmarkCase):
  def get_program(self) -> rasp.SOp:
    return make_next_prime()

  def get_task_description(self) -> str:
    return "Replaces each number with the next prime after that number."

  def get_vocab(self) -> Set:
    return vocabs.get_int_numbers_vocab(max=30)


def is_prime(num):
  """Check if a number is prime."""
  if num <= 1:
    return False
  for i in range(2, int(num ** 0.5) + 1):
    if num % i == 0:
      return False
  return True


def next_prime(n):
  """Return the next highest prime number after n."""
  # Start checking from the next number
  prime_candidate = n
  while True:
    if is_prime(prime_candidate):
      return prime_candidate
    prime_candidate += 1


def make_next_prime():
  return rasp.Map(next_prime, rasp.tokens)