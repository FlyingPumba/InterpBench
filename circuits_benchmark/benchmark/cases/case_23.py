from typing import Set

from circuits_benchmark.benchmark import vocabs
from circuits_benchmark.benchmark.benchmark_case import BenchmarkCase
from tracr.rasp import rasp


class Case23(BenchmarkCase):
  def get_program(self) -> rasp.SOp:
    return make_palindrome_word_spotter(rasp.tokens)

  def get_vocab(self) -> Set:
    return vocabs.get_words_vocab().union({"racecar", "noon"})


def make_palindrome_word_spotter(sop: rasp.SOp) -> rasp.SOp:
    """
    Spots palindrome words in a sequence.

    Example usage:
      palindrome_spotter = make_palindrome_word_spotter(rasp.tokens)
      palindrome_spotter(["racecar", "hello", "noon"])
      >> ["racecar", None, "noon"]
    """
    is_palindrome = rasp.Map(lambda x: x if x == x[::-1] else None, sop)
    return is_palindrome
