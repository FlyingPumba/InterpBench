from typing import Set

from tracr.rasp import rasp

from circuits_benchmark.benchmark import vocabs
from circuits_benchmark.benchmark.common_programs import make_unique_token_extractor
from circuits_benchmark.benchmark.tracr_benchmark_case import TracrBenchmarkCase


class Case21(TracrBenchmarkCase):
  def get_program(self) -> rasp.SOp:
    return make_unique_token_extractor(rasp.tokens)

  def get_task_description(self) -> str:
    return "Extract unique tokens from a string"

  def get_vocab(self) -> Set:
    return vocabs.get_ascii_letters_vocab(count=3)