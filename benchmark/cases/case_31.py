from typing import Set

from benchmark import vocabs
from benchmark.benchmark_case import BenchmarkCase
from tracr.rasp import rasp


class Case31(BenchmarkCase):
  def get_program(self) -> rasp.SOp:
    return make_token_anagram_identifier(rasp.tokens, "listen")

  def get_vocab(self) -> Set:
    return vocabs.get_words_vocab().union({"listen"})


def make_token_anagram_identifier(sop: rasp.SOp, target: str) -> rasp.SOp:
    """
    Identifies if tokens in the sequence are anagrams of a given target word.

    Example usage:
      anagram_identifier = make_token_anagram_identifier(rasp.tokens, "listen")
      anagram_identifier(["enlist", "google", "inlets", "banana"])
      >> [True, False, True, False]
    """
    sorted_target = sorted(target)
    anagram_identifier = rasp.Map(
        lambda x: sorted(x) == sorted_target, sop)
    return anagram_identifier
