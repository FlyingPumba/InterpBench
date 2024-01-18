from typing import Set

from benchmark import vocabs
from benchmark.benchmark_case import BenchmarkCase
from benchmark.common_programs import make_sort
from tracr.rasp import rasp


class Case00009(BenchmarkCase):
  def get_program(self) -> rasp.SOp:
    return make_sort(rasp.tokens, rasp.tokens, 10, 1)

  def get_vocab(self) -> Set:
    return vocabs.get_int_digits_vocab()
