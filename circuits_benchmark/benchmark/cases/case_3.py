from typing import Set

from circuits_benchmark.benchmark import vocabs
from circuits_benchmark.benchmark.benchmark_case import BenchmarkCase
from circuits_benchmark.benchmark.common_programs import make_frac_prevs
from tracr.rasp import rasp


class Case3(BenchmarkCase):
  def get_program(self) -> rasp.SOp:
    is_x = (rasp.tokens == "x").named("is_x")
    return make_frac_prevs(is_x)

  def get_task_description(self) -> str:
    return "Returns the fraction of 'x' in the input up to the i-th position for all i."

  def get_vocab(self) -> Set:
    some_letters = vocabs.get_ascii_letters_vocab(count=3)
    some_letters.add("x")
    return some_letters

  def get_max_seq_len(self) -> int:
    return 5

