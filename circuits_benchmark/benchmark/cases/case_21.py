from typing import Set

from circuits_benchmark.benchmark import vocabs
from circuits_benchmark.benchmark.benchmark_case import BenchmarkCase
from circuits_benchmark.benchmark.common_programs import make_unique_token_extractor
from tracr.rasp import rasp


class Case21(BenchmarkCase):
  def get_program(self) -> rasp.SOp:
    return make_unique_token_extractor(rasp.tokens)

  def get_task_description(self) -> str:
    return "Extract unique tokens from a string"

  def get_vocab(self) -> Set:
    return vocabs.get_ascii_letters_vocab(count=3)