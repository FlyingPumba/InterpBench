from typing import Set

from tracr.rasp import rasp

from circuits_benchmark.benchmark import vocabs
from circuits_benchmark.benchmark.common_programs import shift_by
from circuits_benchmark.benchmark.tracr_benchmark_case import TracrBenchmarkCase


class Case13(TracrBenchmarkCase):
    def get_program(self) -> rasp.SOp:
        return make_token_trend_analysis(rasp.tokens)

    def get_task_description(self) -> str:
        return "Analyzes the trend (increasing, decreasing, constant) of numeric tokens."

    def supports_causal_masking(self) -> bool:
        return False

    def get_vocab(self) -> Set:
        return vocabs.get_int_digits_vocab(count=3)

    def supports_causal_masking(self) -> bool:
        return False


def make_token_trend_analysis(sop: rasp.SOp) -> rasp.SOp:
    """
    Analyzes the trend (increasing, decreasing, constant) of numeric tokens.

    Example usage:
      trend_analysis = make_token_trend_analysis(rasp.tokens)
      trend_analysis([1, 2, 3, 3, 2, 1])
      >> ["increasing", "increasing", "constant", "decreasing", "decreasing", None]
    """
    next_token = shift_by(-1, sop)  # [2, 3, 3, 2, 1, None]

    def second_part_fn(curr, next):
        if curr < next:
            return "increasing"
        elif curr > next:
            return "decreasing"
        else:
            return "constant"

    # Compare the current token with the next token to produce the trend analysis.
    # Curr: [1, 2, 3, 3, 2, 1]
    # Next: [2, 3, 3, 2, 1, None]
    # Result: ["increasing", "increasing", "constant", "decreasing", "decreasing", None]
    trend_analysis = rasp.SequenceMap(second_part_fn, sop, next_token)

    return trend_analysis
