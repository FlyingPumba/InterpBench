from typing import Set

from circuits_benchmark.benchmark import vocabs
from circuits_benchmark.benchmark.common_programs import detect_pattern
from circuits_benchmark.benchmark.tracr_benchmark_case import TracrBenchmarkCase
from tracr.rasp import rasp


class Case12(TracrBenchmarkCase):
  def get_program(self) -> rasp.SOp:
    return detect_pattern(rasp.tokens, "abc")

  def get_task_description(self) -> str:
    return "Detect the pattern 'abc' in the input string."

  def get_vocab(self) -> Set:
    return vocabs.get_ascii_letters_vocab(count=3)

  def get_max_seq_len(self) -> int:
    return 15
