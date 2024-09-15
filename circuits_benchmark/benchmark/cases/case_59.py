from typing import Set, Sequence

from circuits_benchmark.benchmark import vocabs
from circuits_benchmark.benchmark.tracr_benchmark_case import TracrBenchmarkCase
from tracr.rasp import rasp


class Case59(TracrBenchmarkCase):
    def get_program(self) -> rasp.SOp:
        return make_sorting()

    def get_task_description(self) -> str:
        return "Sorts the sequence."

    def get_vocab(self) -> Set:
        return vocabs.get_int_numbers_vocab()

    def supports_causal_masking(self) -> bool:
        return False


def make_sorting() -> rasp.SOp:
    # Create unique keys by combining each element with its index
    # This ensures that even duplicate values can be sorted correctly
    unique_keys = rasp.SequenceMap(lambda x, i: x + i * 0.00001, rasp.tokens, rasp.indices).named("unique_keys")

    # Create a selector that identifies where each unique key is less than every other unique key
    lt_selector = rasp.Select(unique_keys, unique_keys, rasp.Comparison.LT).named("lt_selector")

    # Count the number of elements that each unique key is less than
    # This count determines the sorted position of each element in the output sequence
    sorted_position = rasp.SelectorWidth(lt_selector).named("sorted_position")

    # Place each element into its sorted position by matching each element's sort position with the output sequence's indices
    sorted_sequence_selector = rasp.Select(sorted_position, rasp.indices, rasp.Comparison.EQ).named(
        "sorted_sequence_selector")
    sorted_sequence = rasp.Aggregate(sorted_sequence_selector, rasp.tokens).named("sorted_sequence")

    return sorted_sequence
