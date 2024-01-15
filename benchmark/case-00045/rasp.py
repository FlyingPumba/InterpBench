from tracr.rasp import rasp


def get_program() -> rasp.SOp:
  return make_word_count_by_length(rasp.tokens)

def make_word_count_by_length(sop: rasp.SOp) -> rasp.SOp:
    """
    Counts the number of words in a sequence based on their length.

    Example usage:
      word_count = make_word_count_by_length(rasp.tokens)
      word_count(["apple", "pear", "banana"])
      >> {5: 2, 4: 1}
    """
    word_length = rasp.Map(lambda x: len(x), sop)
    length_selector = rasp.Select(word_length, word_length, rasp.Comparison.EQ)
    word_count = rasp.Aggregate(length_selector, word_length, default=None)
    return word_count