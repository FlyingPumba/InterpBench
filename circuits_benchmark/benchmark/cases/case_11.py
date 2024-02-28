from typing import Set

from circuits_benchmark.benchmark import vocabs
from circuits_benchmark.benchmark.benchmark_case import BenchmarkCase
from circuits_benchmark.benchmark.common_programs import shift_by
from tracr.rasp import rasp


class Case11(BenchmarkCase):
  def get_program(self) -> rasp.SOp:
    return shift_by(2, rasp.tokens)

  def get_vocab(self) -> Set:
    return vocabs.get_str_digits_vocab()