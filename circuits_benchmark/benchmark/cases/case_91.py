from typing import Set

from tracr.rasp import rasp

from circuits_benchmark.benchmark import vocabs
from circuits_benchmark.benchmark.tracr_benchmark_case import TracrBenchmarkCase


class Case91(TracrBenchmarkCase):
    def get_program(self) -> rasp.SOp:
        return make_apply_threshold()

    def get_task_description(self) -> str:
        return "Set all values below a threshold to 0"

    def get_vocab(self) -> Set:
        return vocabs.get_int_numbers_vocab()


def make_apply_threshold(threshold=3) -> rasp.SOp:
    apply_threshold_operation = rasp.Map(lambda x: 0 if x < threshold else x, rasp.tokens).named("apply_threshold")

    return apply_threshold_operation
