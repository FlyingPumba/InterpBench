from typing import Set

from tracr.rasp import rasp

from circuits_benchmark.benchmark import vocabs
from circuits_benchmark.benchmark.tracr_benchmark_case import TracrBenchmarkCase


class Case83(TracrBenchmarkCase):
    def get_program(self) -> rasp.SOp:
        return make_triple()

    def get_task_description(self) -> str:
        return "Triple each element in the sequence."

    def get_vocab(self) -> Set:
        return vocabs.get_int_numbers_vocab()


def make_triple() -> rasp.SOp:
    # Define a lambda function that triples the value of its input.
    triple_func = lambda x: x * 3

    # Apply the triple_func to each element of the sequence using Map.
    triple_sequence = rasp.Map(triple_func, rasp.tokens).named("triple_sequence")

    # Return the SOp that triples each element in the sequence.
    return triple_sequence
