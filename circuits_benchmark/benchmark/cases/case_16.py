from typing import Set

from circuits_benchmark.benchmark import vocabs
from circuits_benchmark.benchmark.benchmark_case import BenchmarkCase
from circuits_benchmark.benchmark.common_programs import make_unique_token_extractor, make_length
from tracr.rasp import rasp


class Case16(BenchmarkCase):
  def get_program(self) -> rasp.SOp:
    return make_lexical_density_calculator(rasp.tokens)

  def get_vocab(self) -> Set:
    return vocabs.get_words_vocab()


def make_lexical_density_calculator(sop: rasp.SOp) -> rasp.SOp:
    """
    Calculates the lexical density of a text (unique words to total words ratio).

    Example usage:
      lexical_density = make_lexical_density_calculator(rasp.tokens)
      lexical_density(["the", "quick", "brown", "the"])
      >> [0.75, 0.75, 0.75, 0]
    """
    unique_words = make_unique_token_extractor(sop)
    total_words = make_length()
    unique_word_count = rasp.SelectorWidth(rasp.Select(unique_words, unique_words,
                                                       lambda key, query: key is not None and query is not None)).named("unique_word_count")
    # note map only works for one input, so we have to use Select if we want the lambda x,y
    temp = rasp.SequenceMap(lambda x, y: (x / y) if y > 0 else None, unique_word_count, total_words)
    lexical_density = rasp.Map(lambda x: x if x is not None else 0, temp)

    return lexical_density
