from typing import Set

from tracr.rasp import rasp

from circuits_benchmark.benchmark import vocabs
from circuits_benchmark.benchmark.common_programs import make_length
from circuits_benchmark.benchmark.tracr_benchmark_case import TracrBenchmarkCase


class Case126(TracrBenchmarkCase):
  def get_program(self) -> rasp.SOp:
    return make_set_to_median()

  def get_task_description(self) -> str:
    return "Replaces each element with the median of all elements."

  def get_vocab(self) -> Set:
    return vocabs.get_int_numbers_vocab()

  def supports_causal_masking(self) -> bool:
    return False


def make_sort() -> rasp.SOp:
  unique_tokens = rasp.SequenceMap(lambda x, y: x + 0.0001 * y, rasp.tokens, rasp.indices)
  smaller = rasp.Select(unique_tokens, unique_tokens, rasp.Comparison.LT).named("smaller")
  target_pos = rasp.SelectorWidth(smaller).named("target_pos")
  sel_new = rasp.Select(target_pos, rasp.indices, rasp.Comparison.EQ)
  return rasp.Aggregate(sel_new, rasp.tokens).named("sort")


def make_set_to_median() -> rasp.SOp:
  sorted_sequence = make_sort()

  length = make_length()
  # Assuming a maximum sequence length to pre-calculate possible indices for median
  mid_index1 = rasp.Map(lambda x: (x - 1) // 2, length).named("mid_index1")
  mid_index2 = rasp.Map(lambda x: x // 2, length).named("mid_index2")

  # Selectors for extracting potential median values
  median_selector1 = rasp.Select(rasp.indices, mid_index1, rasp.Comparison.EQ).named("median_selector1")
  median_selector2 = rasp.Select(rasp.indices, mid_index2, rasp.Comparison.EQ).named("median_selector2")

  # Extracting potential median values
  potential_median1 = rasp.Aggregate(median_selector1, sorted_sequence).named("potential_median1")
  potential_median2 = rasp.Aggregate(median_selector2, sorted_sequence).named("potential_median2")

  # Calculating the average of the two potential medians (handles both odd and even length cases)
  median = rasp.SequenceMap(lambda x, y: (x + y) / 2, potential_median1, potential_median2).named("median")

  return median
