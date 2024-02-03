from typing import Set

from benchmark import vocabs
from benchmark.benchmark_case import BenchmarkCase
from benchmark.common_programs import make_length
from tracr.rasp import rasp


class Case27(BenchmarkCase):
  def get_program(self) -> rasp.SOp:
    return make_token_positional_balance_analyzer(rasp.tokens)

  def get_vocab(self) -> Set:
    return vocabs.get_ascii_letters_vocab(count=5)


def make_token_positional_balance_analyzer(sop: rasp.SOp) -> rasp.SOp:
    """
    Analyzes whether tokens are more towards the start ('front'), end ('rear'), or balanced ('center').

    Example usage:
      balance_analyzer = make_token_positional_balance_analyzer(rasp.tokens)
      balance_analyzer(["a", "b", "c", "d", "e"])
      >> ["front", "front", "center", "rear", "rear"]
    """
    position = rasp.indices
    total_length = make_length()
    balance = rasp.SequenceMap(
        lambda pos, length: "front" if pos < length / 3 else ("rear" if pos > 2 * length / 3 else "center"),
        position, total_length)
    return balance
