from typing import Set

from tracr.rasp import rasp

from circuits_benchmark.benchmark import vocabs
from circuits_benchmark.benchmark.tracr_benchmark_case import TracrBenchmarkCase


class Case48(TracrBenchmarkCase):
    def get_program(self) -> rasp.SOp:
        return make_increment_by_index()

    def get_task_description(self) -> str:
        return "Increments each element by its index."

    def get_vocab(self) -> Set:
        return vocabs.get_int_numbers_vocab()


def make_increment_by_index() -> rasp.SOp:
    # This operation adds each element of the input sequence to its corresponding index.
    incremented_sequence = rasp.SequenceMap(lambda x, y: x + y, rasp.tokens, rasp.indices).named("incremented_sequence")

    return incremented_sequence
