from typing import Set

from tracr.rasp import rasp

from circuits_benchmark.benchmark import vocabs
from circuits_benchmark.benchmark.common_programs import make_pair_balance
from circuits_benchmark.benchmark.tracr_benchmark_case import TracrBenchmarkCase


class Case4(TracrBenchmarkCase):
    def get_program(self) -> rasp.SOp:
        return make_pair_balance(rasp.tokens, "(", ")")

    def get_task_description(self) -> str:
        return "Return fraction of previous open tokens minus the fraction of close tokens."

    def get_vocab(self) -> Set:
        return vocabs.get_ascii_letters_vocab(count=3).union({"(", ")"})
