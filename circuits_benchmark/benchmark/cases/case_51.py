from typing import Set, Sequence

from circuits_benchmark.benchmark import vocabs
from circuits_benchmark.benchmark.tracr_benchmark_case import TracrBenchmarkCase
from tracr.rasp import rasp


class Case51(TracrBenchmarkCase):
    def get_program(self) -> rasp.SOp:
        return make_check_fibonacci()

    def get_task_description(self) -> str:
        return "Checks if each element is a Fibonacci number"

    def get_vocab(self) -> Set:
        return vocabs.get_int_numbers_vocab(min=0, max=100)


def make_check_fibonacci() -> rasp.SOp:
    # Assume a pre-generated Fibonacci sequence up to a certain limit.
    # In practice, this would need to be dynamically generated or sufficiently large.
    fib_sequence = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610]

    # Function to check if a number is in the Fibonacci sequence
    is_fib = lambda x: 1 if x in fib_sequence else 0

    # Apply the check to each element of the input sequence
    check_fibonacci_map = rasp.Map(is_fib, rasp.tokens).named("check_fibonacci_map")

    return check_fibonacci_map
