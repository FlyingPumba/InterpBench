from typing import Set

from tracr.rasp import rasp

from circuits_benchmark.benchmark import vocabs
from circuits_benchmark.benchmark.tracr_benchmark_case import TracrBenchmarkCase


class Case120(TracrBenchmarkCase):
    def get_program(self) -> rasp.SOp:
        return make_flip_halves()

    def get_task_description(self) -> str:
        return "Flips the order of the first and second half of the sequence."

    def get_vocab(self) -> Set:
        return vocabs.get_int_numbers_vocab()

    def supports_causal_masking(self) -> bool:
        return False


def make_flip_halves() -> rasp.SOp:
    len = rasp.SelectorWidth(rasp.Select(rasp.indices, rasp.indices, rasp.Comparison.TRUE))
    half = rasp.Map(lambda x: x / 2, len)
    new_positions = rasp.SequenceMap(lambda x, y: (x - y if x >= y else x + y) if y == int(y) else (
        x if x + 0.5 == y else (x + int(y) + 1 if x < y else x - int(y) - 1)), rasp.indices, half)
    shifter = rasp.Select(new_positions, rasp.indices, rasp.Comparison.EQ)
    return rasp.Aggregate(shifter, rasp.tokens)
