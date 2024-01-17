from typing import Set

from benchmark import vocabs
from benchmark.common_programs import make_sort
from benchmark.defaults import default_max_seq_len
from tracr.rasp import rasp


def get_program() -> rasp.SOp:
  return make_token_sorting_by_length(rasp.tokens)

def make_token_sorting_by_length(sop: rasp.SOp) -> rasp.SOp:
    """
    Sorts tokens in a sequence by their length.

    Example usage:
      token_sort_len = make_token_sorting_by_length(rasp.tokens)
      token_sort_len(["word", "a", "is", "sequence"])
      >> ["a", "is", "word", "sequence"]
    """
    token_length = rasp.Map(lambda x: len(x), sop).named("token_length")
    sorted_tokens = make_sort(sop, token_length, max_seq_len=default_max_seq_len, min_key=1)
    return sorted_tokens


def get_vocab() -> Set:
    return vocabs.get_words_vocab()
