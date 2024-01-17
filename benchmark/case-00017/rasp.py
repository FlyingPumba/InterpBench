from typing import Set

from benchmark import vocabs
from tracr.rasp import rasp
from benchmark.common_programs import make_reverse


def get_program() -> rasp.SOp:
  return make_palindrome_detection(rasp.tokens)

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


def get_vocab() -> Set:
  return vocabs.get_ascii_letters_vocab(count=3)