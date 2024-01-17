from typing import Set

from benchmark import vocabs
from tracr.rasp import rasp
from benchmark.common_programs import shift_by


def get_program() -> rasp.SOp:
  return make_token_rotation_identifier(rasp.tokens, 2)

def make_token_rotation_identifier(sop: rasp.SOp, rotation: int) -> rasp.SOp:
    """
    Identifies if tokens are rotations of each other by a specified number.

    Example usage:
      rotation_identifier = make_token_rotation_identifier(rasp.tokens, 2)
      rotation_identifier(["hello", "llohe", "lohel"])
      >> [True, True, True]
    """
    rotated_token = shift_by(rotation, sop)
    rotation_identifier = rasp.SequenceMap(lambda x, y: x == y, sop, rotated_token)
    return rotation_identifier


def get_vocab() -> Set:
  return vocabs.get_words_vocab().union({"hello", "llohe", "lohel"})