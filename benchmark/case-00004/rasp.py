from typing import Set

from benchmark import vocabs
from benchmark.benchmark_case import BenchmarkCase
from benchmark.common_programs import make_pair_balance
from tracr.rasp import rasp


class Case00004(BenchmarkCase):
  def get_program(self) -> rasp.SOp:
    return make_pair_balance(rasp.tokens, "(", ")")

  def get_vocab(self) -> Set:
    return vocabs.get_ascii_letters_vocab(count=3).union({"(", ")"})