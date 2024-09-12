from typing import Set

from tracr.rasp import rasp

from circuits_benchmark.benchmark import vocabs
from circuits_benchmark.benchmark.tracr_benchmark_case import TracrBenchmarkCase


class Case110(TracrBenchmarkCase):
    def get_program(self) -> rasp.SOp:
        return make_insert_zeros()

    def get_task_description(self) -> str:
        return "Inserts zeros between each element, removing the latter half of the list."

    def get_vocab(self) -> Set:
        return vocabs.get_int_numbers_vocab()

    def supports_causal_masking(self) -> bool:
        return False


def make_insert_zeros() -> rasp.SOp:
    shifter = rasp.Select(rasp.indices, rasp.indices, lambda x, y: x == int(y / 2))
    shifted = rasp.Aggregate(shifter, rasp.tokens)
    return rasp.SequenceMap(lambda x, y: x if y % 2 == 0 else 0, shifted, rasp.indices)
