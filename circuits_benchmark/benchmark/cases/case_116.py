from typing import Set

from tracr.rasp import rasp

from circuits_benchmark.benchmark import vocabs
from circuits_benchmark.benchmark.tracr_benchmark_case import TracrBenchmarkCase


class Case116(TracrBenchmarkCase):
    def get_program(self) -> rasp.SOp:
        return make_check_multiple_of_first()

    def get_task_description(self) -> str:
        return "Checks if each element in a sequence is a multiple of the first one."

    def get_vocab(self) -> Set:
        return vocabs.get_int_numbers_vocab()

    def supports_causal_masking(self) -> bool:
        return False

    def get_max_seq_len(self) -> int:
        return len(self.get_vocab())


def make_check_multiple_of_first():
    first = rasp.Aggregate(rasp.Select(rasp.indices, rasp.Map(lambda x: 0, rasp.indices), rasp.Comparison.EQ),
                           rasp.tokens)
    return rasp.SequenceMap(lambda x, y: (1 if x % y == 0 else 0) if y != 0 else 0, rasp.tokens, first)
