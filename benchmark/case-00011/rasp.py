from typing import Set

from benchmark import vocabs
from benchmark.benchmark_case import BenchmarkCase
from benchmark.common_programs import shift_by
from tracr.rasp import rasp


class Case00011(BenchmarkCase):
  def get_program(self) -> rasp.SOp:
    return shift_by(2, rasp.tokens)

  def get_vocab(self) -> Set:
    return vocabs.get_str_digits_vocab()