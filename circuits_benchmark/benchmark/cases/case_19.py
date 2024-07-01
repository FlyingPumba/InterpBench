from typing import Set

from circuits_benchmark.benchmark import vocabs
from circuits_benchmark.benchmark.common_programs import shift_by
from circuits_benchmark.benchmark.tracr_benchmark_case import TracrBenchmarkCase
from tracr.rasp import rasp


class Case19(TracrBenchmarkCase):
  def get_program(self) -> rasp.SOp:
    return make_sequential_duplicate_removal(rasp.tokens)

  def get_task_description(self) -> str:
    return "Removes consecutive duplicate tokens from a sequence."

  def get_vocab(self) -> Set:
    return vocabs.get_ascii_letters_vocab(count=3)

  def get_max_seq_len(self) -> int:
    return 15


def make_sequential_duplicate_removal(sop: rasp.SOp) -> rasp.SOp:
    """
    Removes consecutive duplicate tokens from a sequence.

    Example usage:
      duplicate_remove = make_sequential_duplicate_removal(rasp.tokens)
      duplicate_remove("aabbcc")
      >> ['a', None, 'b', None, 'c', None]

    Args:
      sop: SOp representing the sequence to process.

    Returns:
      A SOp that maps an input sequence to another sequence where immediate 
      duplicate occurrences of any token are removed.
    """
    shifted_sop = shift_by(1, sop)
    duplicate_removal_sop = rasp.SequenceMap(
        lambda x, y: x if x != y else None, sop, shifted_sop).named("sequential_duplicate_removal")
    return duplicate_removal_sop
