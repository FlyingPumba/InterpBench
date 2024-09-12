from typing import Set

from tracr.rasp import rasp

from circuits_benchmark.benchmark import vocabs
from circuits_benchmark.benchmark.tracr_benchmark_case import TracrBenchmarkCase


class Case26(TracrBenchmarkCase):
  def get_program(self) -> rasp.SOp:
    return make_token_cascade(rasp.tokens)

  def get_task_description(self) -> str:
    return "Creates a cascading effect by repeating each token in sequence incrementally."

  def get_vocab(self) -> Set:
    return vocabs.get_ascii_letters_vocab(count=3)

  def is_trivial(self) -> bool:
      return True


def make_token_cascade(sop: rasp.SOp) -> rasp.SOp:
    """
    Creates a cascading effect by repeating each token in sequence incrementally.

    Example usage:
      token_cascade = make_token_cascade(rasp.tokens)
      token_cascade(["a", "b", "c"])
      >> ["a", "bb", "ccc"]
    """
    cascade_sop = rasp.SequenceMap(lambda x, i: x * (i + 1), sop, rasp.indices)
    return cascade_sop
