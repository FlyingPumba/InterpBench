from typing import Set

from benchmark import vocabs
from benchmark.benchmark_case import BenchmarkCase
from benchmark.common_programs import make_hist
from tracr.rasp import rasp


class Case00007(BenchmarkCase):
  def get_program(self) -> rasp.SOp:
    return make_hist()

  def get_vocab(self) -> Set:
    return vocabs.get_ascii_letters_vocab(count=3)