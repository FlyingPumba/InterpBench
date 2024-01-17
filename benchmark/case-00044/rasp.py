from typing import Set

from benchmark import vocabs
from tracr.rasp import rasp
from benchmark.common_programs import make_unique_token_extractor, make_length


def get_program() -> rasp.SOp:
  return make_lexical_density_calculator(rasp.tokens)

def make_lexical_density_calculator(sop: rasp.SOp) -> rasp.SOp:
    """
    Calculates the lexical density of a text (unique words to total words ratio).

    Example usage:
      lexical_density = make_lexical_density_calculator(rasp.tokens)
      lexical_density(["the", "quick", "brown", "fox"])
      >> 0.75
    """
    unique_words = make_unique_token_extractor(sop)
    total_words = make_length()
    unique_word_count = rasp.SelectorWidth(rasp.Select(unique_words, unique_words, rasp.Comparison.TRUE))
    # note map only works for one input, so we have to use Select if we want the lambda x,y
    temp = rasp.SequenceMap(lambda x, y: (x / y) if y > 0 else None, unique_word_count, total_words)
    lexical_density = rasp.Map(lambda x: x if x is not None else 0, temp)

    return lexical_density


def get_vocab() -> Set:
  return vocabs.get_words_vocab()