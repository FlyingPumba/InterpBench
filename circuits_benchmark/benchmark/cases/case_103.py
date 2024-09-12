from typing import Set

from tracr.rasp import rasp

from circuits_benchmark.benchmark import vocabs
from circuits_benchmark.benchmark.tracr_benchmark_case import TracrBenchmarkCase


class Case103(TracrBenchmarkCase):
    def get_program(self) -> rasp.SOp:
        return make_swap_consecutive()

    def get_task_description(self) -> str:
        return "Swap consecutive numbers in a list"

    def get_vocab(self) -> Set:
        return vocabs.get_int_numbers_vocab()

    def supports_causal_masking(self) -> bool:
        return False


def make_swap_consecutive() -> rasp.SOp:
    len = rasp.SelectorWidth(rasp.Select(rasp.indices, rasp.indices, rasp.Comparison.TRUE))
    swaper = rasp.SequenceMap(lambda x, y: x if (x == y - 1 and y % 2 == 1) else (x + 1 if x % 2 == 0 else x - 1),
                              rasp.indices, len)
    swap_selector = rasp.Select(rasp.indices, swaper, rasp.Comparison.EQ)
    swaped = rasp.Aggregate(swap_selector, rasp.tokens)
    return swaped
