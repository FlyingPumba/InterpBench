from typing import Set

from circuits_benchmark.benchmark import vocabs
from circuits_benchmark.benchmark.common_programs import make_reverse
from circuits_benchmark.benchmark.tracr_benchmark_case import TracrBenchmarkCase
from tracr.rasp import rasp


class Case17(TracrBenchmarkCase):
  def get_program(self) -> rasp.SOp:
    return make_palindrome_detection(rasp.tokens)

  def get_task_description(self) -> str:
    return "Detect if input sequence is a palindrome."

  def supports_causal_masking(self) -> bool:
    return False

  def get_vocab(self) -> Set:
    return vocabs.get_ascii_letters_vocab(count=3)


def make_palindrome_detection(sop: rasp.SOp) -> rasp.SOp:
    """
    Detects palindromes in a sequence of characters.

    Example usage:
      palindrome_detect = make_palindrome_detection(rasp.tokens)
      palindrome_detect("racecar")
      >> [False, False, False, True, False, False, False]

    Args:
      sop: SOp representing the sequence to analyze.

    Returns:
      A SOp that maps an input sequence to a boolean sequence, where True 
      indicates a palindrome at that position.
    """
    reversed_sop = make_reverse(sop)
    palindrome_sop = rasp.SequenceMap(
        lambda x, y: x == y, sop, reversed_sop).named("palindrome_detection")
    return palindrome_sop
