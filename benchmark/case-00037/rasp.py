from typing import Set

from benchmark import vocabs
from tracr.rasp import rasp


def get_program() -> rasp.SOp:
  return make_token_reversal_with_exclusion(rasp.tokens, "nochange")

def make_token_reversal_with_exclusion(sop: rasp.SOp, exclude: str) -> rasp.SOp:
    """
    Reverses each token in the sequence except for specified exclusions.

    Example usage:
      token_reversal = make_token_reversal_with_exclusion(rasp.tokens, "nochange")
      token_reversal(["reverse", "this", "nochange"])
      >> ["esrever", "siht", "nochange"]
    """
    reversal = rasp.Map(lambda x: x[::-1] if x != exclude else x, sop)
    return reversal


def get_vocab() -> Set:
  return vocabs.get_words_vocab()