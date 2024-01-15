from tracr.rasp import rasp


def get_program() -> rasp.SOp:
  return make_token_sandwich(rasp.tokens, "-")

def make_token_sandwich(sop: rasp.SOp, filler: rasp.Value) -> rasp.SOp:
    """
    Places a filler token between each pair of tokens in the sequence.

    Example usage:
      token_sandwich = make_token_sandwich(rasp.tokens, "-")
      token_sandwich(["a", "b", "c"])
      >> ["a", "-", "b", "-", "c"]
    """
    filler_sop = rasp.Full(filler)
    alternate_sop = rasp.SequenceMap(lambda x, y: (x, filler) if y is not None else x, sop, filler_sop)
    return alternate_sop