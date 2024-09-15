from typing import Set

from tracr.rasp import rasp

from circuits_benchmark.benchmark import vocabs
from circuits_benchmark.benchmark.tracr_benchmark_case import TracrBenchmarkCase


class Case47(TracrBenchmarkCase):
    def get_program(self) -> rasp.SOp:
        return make_count_frequency()

    def get_task_description(self) -> str:
        return "Counts the frequency of each unique element"

    def get_vocab(self) -> Set:
        return vocabs.get_ascii_letters_vocab(count=10)

    def supports_causal_masking(self) -> bool:
        return False


def make_count_frequency() -> rasp.SOp:
    # Create a comparison matrix where each element is compared to every other element for equality.
    equality_selector = rasp.Select(rasp.tokens, rasp.tokens, rasp.Comparison.EQ).named("equality_selector")

    # Use SelectorWidth to count the frequency of each element based on the equality comparison matrix.
    frequency_count = rasp.SelectorWidth(equality_selector).named("frequency_count")

    # The result is a sequence where each element is replaced by its frequency count.
    return frequency_count
