from typing import Set

from tracr.rasp import rasp

from circuits_benchmark.benchmark import vocabs
from circuits_benchmark.benchmark.tracr_benchmark_case import TracrBenchmarkCase


class Case82(TracrBenchmarkCase):
    def get_program(self) -> rasp.SOp:
        return make_halve_second_half()

    def get_task_description(self) -> str:
        return "Halve the elements in the second half of the sequence."

    def get_vocab(self) -> Set:
        return vocabs.get_int_numbers_vocab()

    def supports_causal_masking(self) -> bool:
        return False


def make_halve_second_half() -> rasp.SOp:
    # Calculate the length of the sequence and divide it by 2 to determine the start of the second half.
    all_true_selector = rasp.Select(rasp.tokens, rasp.tokens, rasp.Comparison.TRUE).named("all_true_selector")
    length = rasp.SelectorWidth(all_true_selector).named("length")
    half_length = rasp.Map(lambda x: x // 2, length).named("half_length")

    # Use Map to create a boolean sequence indicating whether an index is in the second half.
    in_second_half = rasp.SequenceMap(lambda idx, half: idx >= half, rasp.indices, half_length).named("in_second_half")

    # Halve the elements in the second half.
    halved_sequence = rasp.SequenceMap(lambda x, cond: x / 2 if cond else x, rasp.tokens, in_second_half).named(
        "halved_sequence")

    return halved_sequence
