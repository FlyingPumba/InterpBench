from tracr.rasp import rasp


def get_program() -> rasp.SOp:
  return make_token_abbreviation(rasp.tokens)

def make_token_abbreviation(sop: rasp.SOp) -> rasp.SOp:
    """
    Creates abbreviations for each token in the sequence.

    Example usage:
      token_abbreviation = make_token_abbreviation(rasp.tokens)
      token_abbreviation(["international", "business", "machines"])
      >> ["int", "bus", "mac"]
    """
    abbreviation = rasp.Map(lambda x: x[:3] if len(x) > 3 else x, sop)
    return abbreviation