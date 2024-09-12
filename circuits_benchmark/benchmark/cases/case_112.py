from typing import Set

from tracr.rasp import rasp

from circuits_benchmark.benchmark import vocabs
from circuits_benchmark.benchmark.tracr_benchmark_case import TracrBenchmarkCase


class Case112(TracrBenchmarkCase):
    def get_program(self) -> rasp.SOp:
        return make_difference_to_next()

    def get_task_description(self) -> str:
        return "Compute the difference between each element and the next element in the sequence."

    def get_vocab(self) -> Set:
        return vocabs.get_int_numbers_vocab()

    def supports_causal_masking(self) -> bool:
        return False


def make_difference_to_next():
    def shift_by_one() -> rasp.SOp:
        # Define a selector for shifting sequence by one
        len = rasp.SelectorWidth(rasp.Select(rasp.indices, rasp.indices, rasp.Comparison.TRUE))
        shifted_selector = rasp.Select(rasp.indices,
                                       rasp.SequenceMap(lambda x, y: x if x < y - 1 else x - 1, rasp.indices, len),
                                       lambda x, y: x - 1 == y)
        return rasp.Aggregate(shifted_selector, rasp.tokens)

    shifted = shift_by_one()
    return rasp.SequenceMap(lambda x, y: x - y, shifted, rasp.tokens)
