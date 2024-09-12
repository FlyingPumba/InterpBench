from typing import Set

from tracr.rasp import rasp

from circuits_benchmark.benchmark import vocabs
from circuits_benchmark.benchmark.tracr_benchmark_case import TracrBenchmarkCase


class Case34(TracrBenchmarkCase):
    def get_program(self) -> rasp.SOp:
        return make_vowel_consonant_ratio(rasp.tokens)

    def get_task_description(self) -> str:
        return "Calculate the ratio of vowels to consonants in each word."

    def get_vocab(self) -> Set:
        return vocabs.get_words_vocab()


def make_vowel_consonant_ratio(sop: rasp.SOp) -> rasp.SOp:
    """
    Calculates the ratio of vowels to consonants in each token. Deal with 0 denominator by 
    returning infinity.

    Example usage:
      vowel_consonant_ratio = make_vowel_consonant_ratio(rasp.tokens)
      vowel_consonant_ratio(["apple", "sky", "aeiou"])
      >> [2/3, 0/3, inf]
    """

    def calc_ratio(word):
        vowels = sum(c in 'aeiou' for c in word.lower())
        consonants = len(word) - vowels
        return vowels / consonants if consonants != 0 else float('inf')

    ratio_calculator = rasp.Map(calc_ratio, sop)
    return ratio_calculator
