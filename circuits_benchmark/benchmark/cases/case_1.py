from typing import Set

from circuits_benchmark.benchmark import vocabs
from circuits_benchmark.benchmark.benchmark_case import BenchmarkCase
from circuits_benchmark.benchmark.common_programs import shift_by
from tracr.rasp import rasp


class Case1(BenchmarkCase):
  def get_program(self) -> rasp.SOp:
    return make_token_rotation_identifier(rasp.tokens, 2)

  def get_vocab(self) -> Set:
    return vocabs.get_ascii_letters_vocab(count=5)


def make_token_rotation_identifier(sop: rasp.SOp, rotation: int) -> rasp.SOp:
    """
    Identifies if tokens are rotations of each other by a specified number.

    Example usage:
      rotation_identifier = make_token_rotation_identifier(rasp.tokens, 2)
      rotation_identifier(['d', 'e', 'c', 'e', 'e', 'b', 'c', 'c', 'c'])
      >> [None, None, False, True, False, False, False, False, True]
      Because:
        Orig:    ['d', 'e', 'c', 'e', 'e', 'b', 'c', 'c', 'c']
        Shift 2: [_  , _  , 'd', 'e', 'c', 'e', 'e', 'b', 'c']
        Equals:  [_  , _  ,  F ,  T ,  F ,  F ,  F ,  F ,  T ]
    """
    rotated_token = shift_by(rotation, sop)
    rotation_identifier = rasp.SequenceMap(lambda x, y: x == y, sop, rotated_token)
    return rotation_identifier
