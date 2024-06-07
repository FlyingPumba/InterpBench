from typing import Set

from circuits_benchmark.benchmark import vocabs
from circuits_benchmark.benchmark.benchmark_case import BenchmarkCase
from circuits_benchmark.benchmark.common_programs import make_sort
from tracr.rasp import rasp


class Case22(BenchmarkCase):
  def get_program(self) -> rasp.SOp:
    return make_token_sorting_by_length(rasp.tokens)

  def get_task_description(self) -> str:
    return "Sort words in a sequence by their length."

  def supports_causal_masking(self) -> bool:
    return False

  def get_vocab(self) -> Set:
      return vocabs.get_words_vocab()

  def get_max_seq_len(self) -> int:
    return 10


def make_token_sorting_by_length(sop: rasp.SOp) -> rasp.SOp:
    """
    Sorts tokens in a sequence by their length.

    Example usage:
      token_sort_len = make_token_sorting_by_length(rasp.tokens)
      token_sort_len(["word", "a", "is", "sequence"])
      >> ["a", "is", "word", "sequence"]
    """
    token_length = rasp.Map(lambda x: len(x), sop).named("token_length")
    sorted_tokens = make_sort(sop, token_length, max_seq_len=10, min_key=1)
    return sorted_tokens
