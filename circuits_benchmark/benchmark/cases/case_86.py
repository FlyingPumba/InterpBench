from typing import Set

from tracr.rasp import rasp

from circuits_benchmark.benchmark import vocabs
from circuits_benchmark.benchmark.tracr_benchmark_case import TracrBenchmarkCase


class Case86(TracrBenchmarkCase):
    def get_program(self) -> rasp.SOp:
        return make_check_power_of_n()

    def get_task_description(self) -> str:
        return "Check if each element is a power of 2. Return 1 if true, otherwise 0."

    def get_vocab(self) -> Set:
        return vocabs.get_int_numbers_vocab()


def pow_of_n(n, x):
    while x >= 1:
        if x == n:
            return 1
        x /= n
    return 0


def make_check_power_of_n(n=2) -> rasp.SOp:
    # Check if each element is a power of n. Return 1 if true, otherwise 0.
    return rasp.Map(lambda x: pow_of_n(n, x), rasp.tokens).named(f"check_multiple_of_{n}")
