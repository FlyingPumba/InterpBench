from tracr.rasp import rasp


def get_program() -> rasp.SOp:
  return make_token_mirroring(rasp.tokens)

def make_token_mirroring(sop: rasp.SOp) -> rasp.SOp:
    """
    Mirrors each token in the sequence around its central axis.

    Example usage:
      token_mirror = make_token_mirroring(rasp.tokens)
      token_mirror(["abc", "def", "ghi"])
      >> ["cba", "fed", "ihg"]
    """
    mirrored_sop = rasp.Map(lambda x: x[::-1] if x is not None else None, sop)
    return mirrored_sop