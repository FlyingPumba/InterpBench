from typing import Set

from benchmark import vocabs
from benchmark.benchmark_case import BenchmarkCase
from tracr.rasp import rasp


class Case00036(BenchmarkCase):
  def get_program(self) -> rasp.SOp:
    return make_numeric_token_range_filter(rasp.tokens, 10, 50)

  def get_vocab(self) -> Set:
    return vocabs.get_str_numbers_vocab(0, 100)


def make_numeric_token_range_filter(sop: rasp.SOp, min_val: int, max_val: int) -> rasp.SOp:
    """
    Filters numeric tokens in a sequence based on a specified range.

    Example usage:
      range_filter = make_numeric_token_range_filter(rasp.tokens, 10, 50)
      range_filter(["5", "20", "60", "30"])
      >> [None, "20", None, "30"]
    """
    def in_range(token):
        return token if token.isdigit() and min_val <= int(token) <= max_val else None

    range_filter = rasp.Map(in_range, sop)
    return range_filter
