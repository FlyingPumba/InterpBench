from typing import Set

from circuits_benchmark.benchmark import vocabs
from circuits_benchmark.benchmark.common_programs import make_reverse
from circuits_benchmark.benchmark.tracr_benchmark_case import TracrBenchmarkCase
from tracr.rasp import rasp


class Case2(TracrBenchmarkCase):
  def get_program(self) -> rasp.SOp:
    return make_reverse(rasp.tokens)

  def get_task_description(self) -> str:
    return "Reverse the input sequence."

  def get_vocab(self) -> Set:
      return vocabs.get_ascii_letters_vocab()

  def supports_causal_masking(self) -> bool:
    return False
