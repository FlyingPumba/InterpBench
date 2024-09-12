from typing import Set

from tracr.rasp import rasp

from circuits_benchmark.benchmark import vocabs
from circuits_benchmark.benchmark.tracr_benchmark_case import TracrBenchmarkCase


class Case87(TracrBenchmarkCase):
  def get_program(self) -> rasp.SOp:
    return make_binarize()

  def get_task_description(self) -> str:
    return "Binarize a sequence of integers using a threshold."

  def get_vocab(self) -> Set:
    return vocabs.get_int_numbers_vocab()


def make_binarize(threshold=3) -> rasp.SOp:
  compare_to_threshold = rasp.Map(lambda x: x >= threshold, rasp.tokens)
  binarized_sequence = rasp.Map(lambda x: 1 if x else 0, compare_to_threshold)
  return binarized_sequence
