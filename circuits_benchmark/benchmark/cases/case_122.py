from typing import Set

from tracr.rasp import rasp

from circuits_benchmark.benchmark import vocabs
from circuits_benchmark.benchmark.tracr_benchmark_case import TracrBenchmarkCase


class Case122(TracrBenchmarkCase):
    def get_program(self) -> rasp.SOp:
        return make_check_divisibility()

    def get_task_description(self) -> str:
        return "Check if each number is divisible by 3."

    def get_vocab(self) -> Set:
        return vocabs.get_int_numbers_vocab(max=30)


def make_check_divisibility(divisor=3):
    return rasp.Map(lambda x: 1 if x % divisor == 0 else 0, rasp.tokens)
