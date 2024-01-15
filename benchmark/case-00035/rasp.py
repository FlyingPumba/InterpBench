from tracr.rasp import rasp


def get_program() -> rasp.SOp:
  return make_token_capitalization_alternator(rasp.tokens)

def make_token_capitalization_alternator(sop: rasp.SOp) -> rasp.SOp:
    """
    Alternates capitalization of each character in tokens.

    Example usage:
      capitalization_alternator = make_token_capitalization_alternator(rasp.tokens)
      capitalization_alternator(["hello", "world"])
      >> ["HeLlO", "WoRlD"]
    """
    def alternate_capitalization(word):
        return ''.join(c.upper() if i % 2 == 0 else c.lower() for i, c in enumerate(word))

    alternator = rasp.Map(alternate_capitalization, sop)
    return alternator