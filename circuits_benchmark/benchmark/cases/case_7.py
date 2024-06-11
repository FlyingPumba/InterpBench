from typing import Set

from circuits_benchmark.benchmark import vocabs
from circuits_benchmark.benchmark.benchmark_case import BenchmarkCase
from circuits_benchmark.benchmark.common_programs import make_hist
from tracr.rasp import rasp


class Case7(BenchmarkCase):
  def get_program(self) -> rasp.SOp:
    return make_hist()

  def get_task_description(self) -> str:
    return "Returns the number of times each token occurs in the input."

  def supports_causal_masking(self) -> bool:
    return False

  def get_vocab(self) -> Set:
    return vocabs.get_ascii_letters_vocab(count=3)