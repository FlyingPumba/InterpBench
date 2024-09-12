from typing import Set

from tracr.rasp import rasp

from circuits_benchmark.benchmark import vocabs
from circuits_benchmark.benchmark.tracr_benchmark_case import TracrBenchmarkCase


class Case75(TracrBenchmarkCase):
    def get_program(self) -> rasp.SOp:
        return make_element_double()

    def get_task_description(self) -> str:
        return "Double each element of the input sequence."

    def get_vocab(self) -> Set:
        return vocabs.get_int_numbers_vocab()


def make_element_double() -> rasp.SOp:
    # Apply the doubling function to each element of the input sequence.
    return rasp.Map(lambda x: x * 2, rasp.tokens).named("double_elements")
