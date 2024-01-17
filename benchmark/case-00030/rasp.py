from typing import Set

from benchmark import vocabs
from tracr.rasp import rasp


def get_program() -> rasp.SOp:
  return make_numeric_range_tagging(rasp.tokens, 10, 20)

def make_numeric_range_tagging(sop: rasp.SOp, lower_bound: int, upper_bound: int) -> rasp.SOp:
    """
    Tags numeric tokens in a sequence based on whether they fall within a given range.

    Example usage:
      range_tagging = make_numeric_range_tagging(rasp.tokens, 10, 20)
      range_tagging(["5", "15", "25", "20"])
      >> [False, True, False, True]
    """
    range_tagging = rasp.Map(
        lambda x: lower_bound <= int(x) <= upper_bound if x.isdigit() else False, sop)
    return range_tagging


def get_vocab() -> Set:
  return vocabs.get_str_numbers_vocab(min=0, max=30)