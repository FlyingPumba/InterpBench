from tracr.rasp import rasp


def get_program() -> rasp.SOp:
  return make_palindrome_word_spotter(rasp.tokens)

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