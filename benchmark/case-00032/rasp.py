from typing import Set

from benchmark import vocabs
from tracr.rasp import rasp
from benchmark.common_programs import shift_by


def get_program() -> rasp.SOp:
  return make_token_boundary_detector(rasp.tokens)

def make_token_boundary_detector(sop: rasp.SOp) -> rasp.SOp:
    """
    Detects the boundaries between different types of tokens in a sequence.

    Example usage:
      token_boundary = make_token_boundary_detector(rasp.tokens)
      token_boundary(["apple", "banana", "apple", "orange"])
      >> [False, True, False, True]
    """
    previous_token = shift_by(1, sop)
    boundary_detector = rasp.SequenceMap(
        lambda x, y: x != y, sop, previous_token)
    return boundary_detector


def get_vocab() -> Set:
  return vocabs.get_words_vocab()