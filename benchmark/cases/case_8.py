from typing import Set

from benchmark import vocabs
from benchmark.benchmark_case import BenchmarkCase
from benchmark.common_programs import make_sort_unique
from tracr.rasp import rasp


class Case8(BenchmarkCase):
  def get_program(self) -> rasp.SOp:
    return make_sort_unique(rasp.tokens, rasp.tokens)

  def get_vocab(self) -> Set:
    return vocabs.get_str_digits_vocab()