from typing import Set

from circuits_benchmark.benchmark import vocabs
from circuits_benchmark.benchmark.benchmark_case import BenchmarkCase
from circuits_benchmark.benchmark.common_programs import make_reverse
from tracr.rasp import rasp


class Case2(BenchmarkCase):
  def get_program(self) -> rasp.SOp:
    return make_reverse(rasp.tokens)

  def get_vocab(self) -> Set:
      return vocabs.get_ascii_letters_vocab()

  def supports_causal_masking(self) -> bool:
    return False
