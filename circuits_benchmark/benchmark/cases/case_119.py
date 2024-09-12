from typing import Set

from tracr.rasp import rasp

from circuits_benchmark.benchmark import vocabs
from circuits_benchmark.benchmark.tracr_benchmark_case import TracrBenchmarkCase


class Case119(TracrBenchmarkCase):
  def get_program(self) -> rasp.SOp:
    return make_polynomial()

  def get_task_description(self) -> str:
    return "Evaluates a polynomial with sequence elements as parameters. The x is represented by the first entry, the rest are parameters."

  def get_vocab(self) -> Set:
    return vocabs.get_int_numbers_vocab(max=6)

  def get_max_seq_len(self) -> int:
    return 5

  def supports_causal_masking(self) -> bool:
    return False


def make_polynomial(degree=2) -> rasp.SOp:
  """
  Computes the result of a polynomial. The first element of the sequence is treated as the base of the polynomial,
  while the following ones are the weights.
  Example: input [3,2,3,1,4] is treated as the polynomial 2*3^3 + 3*3^2 + 1*3 + 4 = 88
  Note that the degree parameter should correspond to the length of the input sequence minus two.
  """

  aggregator = rasp.tokens - rasp.tokens
  first_element_selector = rasp.Select(rasp.indices, aggregator, rasp.Comparison.EQ).named("first_element_selector")
  base = rasp.Aggregate(first_element_selector, rasp.tokens)

  # Function to create selectors and weights
  def create_elem(i):
    selector = rasp.Select(rasp.indices, rasp.Map(lambda x: i, rasp.tokens), rasp.Comparison.EQ).named(
      f"selector_{i}")
    weight = rasp.Aggregate(selector, rasp.tokens, default=None).named(f"weight_{i}")
    elem = rasp.SequenceMap(lambda x, y: y * (x ** (degree + 1 - i)), base, weight)
    return rasp.SequenceMap(lambda x, y: x + y, aggregator, elem)

  # Applying the function for each term
  for i in range(1, degree + 2):
    aggregator = create_elem(i)

  return aggregator
