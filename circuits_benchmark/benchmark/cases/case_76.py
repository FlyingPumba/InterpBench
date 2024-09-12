from typing import Set

from tracr.rasp import rasp

from circuits_benchmark.benchmark import vocabs
from circuits_benchmark.benchmark.tracr_benchmark_case import TracrBenchmarkCase


class Case76(TracrBenchmarkCase):
    def get_program(self) -> rasp.SOp:
        return make_zero_even_indices()

    def get_task_description(self) -> str:
        return "Set even indices to 0"

    def get_vocab(self) -> Set:
        return vocabs.get_int_numbers_vocab()


def make_zero_even_indices() -> rasp.SOp:
    # Create a sequence of indices
    indices = rasp.Map(lambda x: x, rasp.indices).named("indices")

    # Create a sequence where even indices are marked with 0 and odd indices with -1
    even_odd_marker = rasp.Map(lambda x: 0 if x % 2 == 0 else -1, indices).named("even_odd_marker")

    # Use SequenceMap to combine the original sequence with the marker sequence
    # If the marker is 0, return 0 (for even indices); otherwise, return the original element (for odd indices)
    final_sequence = rasp.SequenceMap(lambda elem, marker: elem if marker == -1 else 0, rasp.tokens,
                                      even_odd_marker).named("final_sequence")

    return final_sequence
