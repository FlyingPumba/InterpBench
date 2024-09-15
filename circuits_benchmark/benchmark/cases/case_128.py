from typing import Set

from tracr.rasp import rasp

from circuits_benchmark.benchmark import vocabs
from circuits_benchmark.benchmark.tracr_benchmark_case import TracrBenchmarkCase


class Case128(TracrBenchmarkCase):
    def get_program(self) -> rasp.SOp:
        return make_swap_first_last()

    def get_task_description(self) -> str:
        return "Swap the first and last elements of a list."

    def get_vocab(self) -> Set:
        return vocabs.get_int_numbers_vocab()

    def supports_causal_masking(self) -> bool:
        return False


def condition(a, b):
    if a == 0:
        return b - 1
    elif a + 1 == b:
        return 0
    else:
        return a


def make_swap_first_last():
    swaper = rasp.SequenceMap(lambda x, y: condition(x, y), rasp.indices,
                              rasp.SelectorWidth(rasp.Select(rasp.indices, rasp.indices, rasp.Comparison.TRUE)))
    swap_selector = rasp.Select(swaper, rasp.indices, rasp.Comparison.EQ)
    return rasp.Aggregate(swap_selector, rasp.tokens)
