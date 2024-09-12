from typing import Set

from tracr.rasp import rasp

from circuits_benchmark.benchmark import vocabs
from circuits_benchmark.benchmark.tracr_benchmark_case import TracrBenchmarkCase


class Case68(TracrBenchmarkCase):
    def get_program(self) -> rasp.SOp:
        return make_increment_to_multiple_of_three()

    def get_task_description(self) -> str:
        return "Increment each element until it becomes a multiple of 3"

    def get_vocab(self) -> Set:
        return vocabs.get_int_numbers_vocab()


def make_increment_to_multiple_of_three() -> rasp.SOp:
    increment_to_multiple_of_three = rasp.Map(lambda x: x + (3 - (x % 3)) % 3, rasp.tokens).named(
        "increment_to_multiple_of_three")

    return increment_to_multiple_of_three
