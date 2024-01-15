from tracr.rasp import rasp
from benchmark.common_programs import make_hist, make_length


def get_program() -> rasp.SOp:
  return make_token_cascade(rasp.tokens)

def make_token_cascade(sop: rasp.SOp) -> rasp.SOp:
    """
    Creates a cascading effect by repeating each token in sequence incrementally.

    Example usage:
      token_cascade = make_token_cascade(rasp.tokens)
      token_cascade(["a", "b", "c"])
      >> ["a", "bb", "ccc"]
    """
    cascade_sop = rasp.SequenceMap(lambda x, i: x * (i + 1), sop, rasp.indices)
    return cascade_sop