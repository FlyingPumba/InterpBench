from typing import Set

from tracr.rasp import rasp

from circuits_benchmark.benchmark import vocabs
from circuits_benchmark.benchmark.common_programs import make_length, make_sort
from circuits_benchmark.benchmark.tracr_benchmark_case import TracrBenchmarkCase


class Case81(TracrBenchmarkCase):
  def get_program(self) -> rasp.SOp:
    return make_compute_median()

  def get_task_description(self) -> str:
    return ""

  def get_vocab(self) -> Set:
    return vocabs.get_int_numbers_vocab()

  def supports_causal_masking(self) -> bool:
    return False


def make_compute_median() -> rasp.SOp:
  # Sort the sequence.
  sorted_sequence = make_sort(rasp.tokens, rasp.tokens, max_seq_len=100, min_key=1)

  # Compute the length of the sequence.
  length = make_length()

  # Compute indices for the middle elements.
  middle1 = rasp.Map(lambda x: (x - 1) // 2, length)
  middle2 = rasp.Map(lambda x: x // 2, length)

  # Select middle elements based on computed indices.
  median1 = rasp.Aggregate(rasp.Select(rasp.indices, middle1, rasp.Comparison.EQ), sorted_sequence)
  median2 = rasp.Aggregate(rasp.Select(rasp.indices, middle2, rasp.Comparison.EQ), sorted_sequence)

  # Compute the average of the two middle elements (handles both odd and even-length sequences).
  median = rasp.SequenceMap(lambda x, y: (x + y) / 2, median1, median2)

  return median
