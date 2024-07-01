from typing import Set

from circuits_benchmark.benchmark import vocabs
from circuits_benchmark.benchmark.common_programs import make_hist, make_length
from circuits_benchmark.benchmark.tracr_benchmark_case import TracrBenchmarkCase
from tracr.rasp import rasp


class Case18(TracrBenchmarkCase):
  def get_program(self) -> rasp.SOp:
    return make_token_frequency_classifier(rasp.tokens)

  def get_task_description(self) -> str:
    return "Classify each token based on its frequency as 'rare', 'common', or 'frequent'."

  def supports_causal_masking(self) -> bool:
    return False

  def get_vocab(self) -> Set:
    return vocabs.get_ascii_letters_vocab(count=5)


def make_token_frequency_classifier(sop: rasp.SOp) -> rasp.SOp:
    """
    Classifies each token based on its frequency as 'rare', 'common', or 'frequent'.

    Example usage:
      frequency_classifier = make_token_frequency_classifier(rasp.tokens)
      frequency_classifier(["a", "b", "a", "c", "a", "b"])
      >> ["frequent", "common", "frequent", "rare", "frequent", "common"]
    """
    frequency = make_hist()
    total_tokens = make_length()
    frequency_classification = rasp.SequenceMap(
        lambda freq, total: "frequent" if freq > total / 2 else ("common" if freq > total / 4 else "rare"),
        frequency, total_tokens)
    return frequency_classification
