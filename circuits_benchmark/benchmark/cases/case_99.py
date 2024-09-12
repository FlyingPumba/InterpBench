from typing import Set

from tracr.rasp import rasp

from circuits_benchmark.benchmark import vocabs
from circuits_benchmark.benchmark.tracr_benchmark_case import TracrBenchmarkCase


class Case99(TracrBenchmarkCase):
    def get_program(self) -> rasp.SOp:
        return make_sum_with_next()

    def get_task_description(self) -> str:
        return "Sum each element with the next one in the sequence."

    def get_vocab(self) -> Set:
        return vocabs.get_int_numbers_vocab()

    def supports_causal_masking(self) -> bool:
        return False


def make_sum_with_next() -> rasp.SOp:
    # Function to shift the sequence by one position
    def shift_by_one() -> rasp.SOp:
        # Define a selector for shifting sequence by one
        len = rasp.SelectorWidth(rasp.Select(rasp.indices, rasp.indices, rasp.Comparison.TRUE))
        shifted_selector = rasp.Select(rasp.indices,
                                       rasp.SequenceMap(lambda x, y: x if x < y - 1 else x - 1, rasp.indices, len),
                                       lambda x, y: x - 1 == y)
        return rasp.Aggregate(shifted_selector, rasp.tokens)

    shifted_sequence = shift_by_one()
    # Add the original sequence to the shifted sequence
    sum_with_next = rasp.SequenceMap(lambda x, y: x + y if y != 0 else x, rasp.tokens, shifted_sequence)

    return sum_with_next
