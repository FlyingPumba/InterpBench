from typing import Set

from benchmark import vocabs
from benchmark.benchmark_case import BenchmarkCase
from tracr.rasp import rasp


class Case20(BenchmarkCase):
  def get_program(self) -> rasp.SOp:
    return make_token_counter(rasp.tokens, "a")

  def get_vocab(self) -> Set:
    return vocabs.get_ascii_letters_vocab(count=3)


def make_token_counter(sop: rasp.SOp, target_token: rasp.Value) -> rasp.SOp:
    """
    Counts occurrences of a specific token in a sequence.

    Example usage:
      token_count = make_token_counter(rasp.tokens, "a")
      token_count("banana")
      >> [1, 1, 1, 1, 1, 1]

    Args:
      sop: SOp representing the sequence to analyze.
      target_token: The token to count occurrences of.

    Returns:
      A SOp that maps an input sequence to a sequence where each element is 
      the count of the target token up to that position.
    """
    token_equals = rasp.Map(lambda x: x == target_token, sop).named("token_equals")
    pre_agg = rasp.Select(rasp.indices, rasp.indices, rasp.Comparison.LEQ).named("pre_agg")
    count_sop = rasp.Aggregate(
        pre_agg,
        token_equals).named("count_sop")
    count_sop = rasp.Map(lambda x: x if x is not None else 0, count_sop).named("count_sop")
    return count_sop
