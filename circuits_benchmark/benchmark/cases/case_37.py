from typing import Set

from tracr.rasp import rasp

from circuits_benchmark.benchmark import vocabs
from circuits_benchmark.benchmark.tracr_benchmark_case import TracrBenchmarkCase


class Case37(TracrBenchmarkCase):
    def get_program(self) -> rasp.SOp:
        return make_token_reversal_with_exclusion(rasp.tokens, "nochange")

    def get_task_description(self) -> str:
        return "Reverses each word in the sequence except for specified exclusions."

    def get_vocab(self) -> Set:
        return vocabs.get_words_vocab()

    def is_trivial(self) -> bool:
        return True


def make_token_reversal_with_exclusion(sop: rasp.SOp, exclude: str) -> rasp.SOp:
    """
    Reverses each token in the sequence except for specified exclusions.

    Example usage:
      token_reversal = make_token_reversal_with_exclusion(rasp.tokens, "nochange")
      token_reversal(["reverse", "this", "nochange"])
      >> ["esrever", "siht", "nochange"]
    """
    reversal = rasp.Map(lambda x: x[::-1] if x != exclude else x, sop)
    return reversal
