from typing import Set

from tracr.rasp import rasp

from circuits_benchmark.benchmark import vocabs
from circuits_benchmark.benchmark.tracr_benchmark_case import TracrBenchmarkCase


class Case44(TracrBenchmarkCase):
    def get_program(self) -> rasp.SOp:
        return make_count_greater_than()

    def get_task_description(self) -> str:
        return "Replaces each element with the number of elements greater than it in the sequence"

    def get_vocab(self) -> Set:
        return vocabs.get_int_numbers_vocab()

    def supports_causal_masking(self) -> bool:
        return False


def make_count_greater_than() -> rasp.SOp:
    # Creating a selector that identifies elements greater than each element.
    greater_than_selector = rasp.Select(rasp.tokens, rasp.tokens, rasp.Comparison.GT).named("greater_than_selector")

    # Counting the number of elements greater than each element.
    count_greater_than = rasp.SelectorWidth(greater_than_selector).named("count_greater_than")

    return count_greater_than
