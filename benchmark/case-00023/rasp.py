from typing import Set

from benchmark import vocabs
from tracr.rasp import rasp
from benchmark.common_programs import shift_by


def get_program() -> rasp.SOp:
  return make_token_pairing(rasp.tokens)

def make_token_pairing(sop: rasp.SOp) -> rasp.SOp:
    """
    Pairs adjacent tokens in a sequence.

    Example usage:
      token_pair = make_token_pairing(rasp.tokens)
      token_pair(["a", "b", "c", "d"])
      >> [("a", "b"), ("b", "c"), ("c", "d"), None]
    """
    shifted_sop = shift_by(1, sop)
    token_pair = rasp.SequenceMap(lambda x, y: (x, y) if y is not None else None, sop, shifted_sop)
    return token_pair


def get_vocab() -> Set:
  return vocabs.get_ascii_letters_vocab(count=3)