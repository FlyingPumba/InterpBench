from typing import Set

from tracr.rasp import rasp

from circuits_benchmark.benchmark import vocabs
from circuits_benchmark.benchmark.tracr_benchmark_case import TracrBenchmarkCase


class Case88(TracrBenchmarkCase):
    def get_program(self) -> rasp.SOp:
        return make_average_first_last()

    def get_task_description(self) -> str:
        return "Calculate the average of the first and last elements of a sequence."

    def get_vocab(self) -> Set:
        return vocabs.get_int_numbers_vocab()

    def supports_causal_masking(self) -> bool:
        return False


def make_average_first_last() -> rasp.SOp:
    # Assuming the first element is at index 0, which is always true
    # Selector for the first element
    first_elem_selector = rasp.Select(rasp.indices, rasp.indices, lambda x, y: x == 0).named("first_elem_selector")
    first_elem = rasp.Aggregate(first_elem_selector, rasp.tokens, default=None).named("first_elem")

    # Assuming the last element can be simulated by reversing the sequence and then selecting the first element
    # Reverse the sequence
    reverser = rasp.SequenceMap(lambda x, y: y - x - 1, rasp.indices,
                                rasp.SelectorWidth(rasp.Select(rasp.indices, rasp.indices, rasp.Comparison.TRUE)))
    reverse_selector = rasp.Select(reverser, rasp.indices, rasp.Comparison.EQ)
    reversed_sequence = rasp.Aggregate(reverse_selector, rasp.tokens).named("reversed_sequence")
    # Select the first element which is effectively the last element of the original sequence
    last_elem = rasp.Aggregate(first_elem_selector, reversed_sequence, default=None).named("last_elem")

    # Calculate the average of the first and last elements
    average_first_last = rasp.SequenceMap(lambda x, y: (x + y) / 2.0, first_elem, last_elem).named("average_first_last")

    return average_first_last
