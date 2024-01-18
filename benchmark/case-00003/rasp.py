from typing import Set

from benchmark import vocabs
from tracr.rasp import rasp
from benchmark.benchmark_case import BenchmarkCase
from benchmark.common_programs import make_frac_prevs


class Case00003(BenchmarkCase):
  def get_program(self) -> rasp.SOp:
    return make_frac_prevs(rasp.tokens == "x")

  def get_vocab(self) -> Set:
    some_letters = vocabs.get_ascii_letters_vocab(count=3)
    some_letters.add("x")
    return some_letters