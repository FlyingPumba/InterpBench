from typing import Set, Sequence

from circuits_benchmark.benchmark import vocabs
from circuits_benchmark.benchmark.tracr_benchmark_case import TracrBenchmarkCase
from tracr.rasp import rasp


class Case53(TracrBenchmarkCase):
    def get_program(self) -> rasp.SOp:
        return make_increment_odd_indices()

    def get_task_description(self) -> str:
        return "Increment elements at odd indices by 1"

    def get_vocab(self) -> Set:
        return vocabs.get_int_numbers_vocab()


def make_increment_odd_indices() -> rasp.SOp:
    # Marks odd indices with 1 and even indices with 0
    odd_index_marker = rasp.Map(lambda x: x % 2, rasp.indices).named("odd_index_marker")

    # Increment elements at odd indices by 1
    incremented_elements = rasp.SequenceMap(
        lambda elem, mark: elem + mark, rasp.tokens, odd_index_marker
    ).named("incremented_elements")

    return incremented_elements
