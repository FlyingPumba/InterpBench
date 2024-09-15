import math
from typing import Set

from tracr.rasp import rasp

from circuits_benchmark.benchmark import vocabs
from circuits_benchmark.benchmark.tracr_benchmark_case import TracrBenchmarkCase


class Case84(TracrBenchmarkCase):
    def get_program(self) -> rasp.SOp:
        return make_arctangent()

    def get_task_description(self) -> str:
        return "Apply the arctangent function to each element of the input sequence."

    def get_vocab(self) -> Set:
        return vocabs.get_int_numbers_vocab()


def make_arctangent() -> rasp.SOp:
    return rasp.Map(lambda x: math.atan(x), rasp.tokens).named("arctangent")
