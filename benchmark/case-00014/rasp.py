from typing import Set

from benchmark import vocabs
from benchmark.benchmark_case import BenchmarkCase
from benchmark.program_evaluation_type import causal_and_regular
from tracr.rasp import rasp


class Case00014(BenchmarkCase):
  def get_program(self) -> rasp.SOp:
    return make_count(rasp.tokens, "a")

  def get_vocab(self) -> Set:
    return vocabs.get_ascii_letters_vocab(count=3)

@causal_and_regular
def make_count(sop, token):
  """Returns the count of `token` in `sop`.

  The output sequence contains this count in each position.

  Example usage:
    count = make_count(tokens, "a")
    count(["a", "a", "a", "b", "b", "c"])
    >> [3, 3, 3, 3, 3, 3]
    count(["c", "a", "b", "c"])
    >> [1, 1, 1, 1]

  Args:
    sop: Sop to count tokens in.
    token: Token to count.
  """
  return rasp.SelectorWidth(rasp.Select(
      sop, sop, lambda k, q: k == token)).named(f"count_{token}")