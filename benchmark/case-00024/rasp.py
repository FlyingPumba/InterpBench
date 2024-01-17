from typing import Set

from benchmark import vocabs
from tracr.rasp import rasp


def get_program() -> rasp.SOp:
  return make_leading_token_identification(rasp.tokens)

def make_leading_token_identification(sop: rasp.SOp) -> rasp.SOp:
    """
    Identifies the first occurrence of each token in a sequence.

    Example usage:
      leading_token_id = make_leading_token_identification(rasp.tokens)
      leading_token_id(["x", "y", "x", "z", "y"])
      >> [True, True, False, True, False]
    """
    first_occurrence = rasp.Aggregate(
        rasp.Select(rasp.indices, rasp.indices, rasp.Comparison.EQ),
        sop, default=None).named("first_occurrence")
    return first_occurrence


def get_vocab() -> Set:
  return vocabs.get_ascii_letters_vocab(count=3)