from typing import Set

from tracr.rasp import rasp

from circuits_benchmark.benchmark import vocabs
from circuits_benchmark.benchmark.tracr_benchmark_case import TracrBenchmarkCase


class Case111(TracrBenchmarkCase):
    def get_program(self) -> rasp.SOp:
        return make_last_element()

    def get_task_description(self) -> str:
        return "Returns the last element of the sequence and pads the rest with zeros."

    def get_vocab(self) -> Set:
        return vocabs.get_int_numbers_vocab()

    def supports_causal_masking(self) -> bool:
        return False


def make_last_element() -> rasp.SOp:
    # Generating the length of the sequence
    length = rasp.SelectorWidth(rasp.Select(rasp.tokens, rasp.tokens, rasp.Comparison.TRUE)).named("length")

    # Selector for the last element based on length minus 1
    last_element_selector = rasp.Select(rasp.indices, rasp.Map(lambda x: x - 1, length), rasp.Comparison.EQ).named(
        "last_element_selector")

    # Broadcasting the last element across the entire sequence
    last_element_sequence = rasp.Aggregate(last_element_selector, rasp.tokens).named("last_element_sequence")

    return rasp.SequenceMap(lambda x, y: x if y == 0 else 0, last_element_sequence, rasp.indices)
