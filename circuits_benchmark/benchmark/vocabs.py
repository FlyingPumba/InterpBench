import random
import string
from typing import Set

TRACR_BOS = "BOS"
TRACR_PAD = "PAD"

def get_ascii_letters_vocab(count=len(string.ascii_lowercase)) -> Set:
  return set(string.ascii_lowercase[:count])


def get_str_digits_vocab(count=len(string.digits)) -> Set:
  return set(string.digits[:count])


def get_int_digits_vocab(count=len(string.digits)) -> Set:
  return set([int(d) for d in get_str_digits_vocab(count=count)])


def get_str_numbers_vocab(min=0, max=20) -> Set:
    return set([str(d) for d in range(min, max)])


def get_int_numbers_vocab(min=0, max=11) -> Set:
    return set([d for d in range(min, max)])


def get_float_numbers_vocab(min=0, max=5, count=20) -> Set:
    return set([min + x * (max - min) / count for x in range(count)])


def get_words_vocab(seed=42, min_chars=1, max_chars=8, min_words=5, max_words=20) -> Set:
    """Generate a set of random words."""
    random.seed(seed)
    vocab: Set = set()
    for _ in range(random.randint(min_words, max_words)):
        word = "".join(random.choice(string.ascii_letters) for _ in range(random.randint(min_chars, max_chars)))
        vocab.add(word)
    return vocab