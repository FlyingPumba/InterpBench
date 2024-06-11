from typing import Set

from circuits_benchmark.benchmark import vocabs
from circuits_benchmark.benchmark.benchmark_case import BenchmarkCase
from circuits_benchmark.benchmark.common_programs import make_hist, make_length
from tracr.rasp import rasp


class Case25(BenchmarkCase):
  def get_program(self) -> rasp.SOp:
    return make_token_frequency_normalization()

  def get_task_description(self) -> str:
    return "Normalizes token frequencies in a sequence to a range between 0 and 1."

  def supports_causal_masking(self) -> bool:
    return False

  def get_vocab(self) -> Set:
    return vocabs.get_ascii_letters_vocab(count=3)


def make_token_frequency_normalization() -> rasp.SOp:
    """
    Normalizes token frequencies in a sequence to a range between 0 and 1.

    Example usage:
      token_freq_norm = make_token_frequency_normalization(rasp.tokens)
      token_freq_norm(["a", "a", "b", "c", "c", "c"])
      >> [0.33, 0.33, 0.16, 0.5, 0.5, 0.5]
    """
    normalized_freq = rasp.SequenceMap(lambda x, y: (x / y) if y > 0 else None, make_hist(), make_length())
    return normalized_freq
