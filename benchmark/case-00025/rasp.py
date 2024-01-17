from typing import Set

from benchmark import vocabs
from tracr.rasp import rasp
from benchmark.common_programs import make_hist, make_length


def get_program() -> rasp.SOp:
  return make_token_frequency_normalization()

def make_token_frequency_normalization() -> rasp.SOp:
    """
    Normalizes token frequencies in a sequence to a range between 0 and 1.

    Example usage:
      token_freq_norm = make_token_frequency_normalization(rasp.tokens)
      token_freq_norm(["a", "a", "b", "c", "c", "c"])
      >> [0.33, 0.33, 0.16, 0.5, 0.5, 0.5]
    """
    normalized_freq = rasp.SequenceMap(lambda x, y: (x / y) if y > 0 else None, make_hist(), make_length())
    return normalized_freq


def get_vocab() -> Set:
  return vocabs.get_ascii_letters_vocab(count=3)