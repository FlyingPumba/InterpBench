from tracr.rasp import rasp
from benchmark.common_programs import shift_by


def get_program() -> rasp.SOp:
  return make_sequential_gap_filler(rasp.tokens, "-")

def make_sequential_gap_filler(sop: rasp.SOp, filler: str) -> rasp.SOp:
    """
    Fills gaps between tokens with a specified filler.

    Example usage:
      gap_filler = make_sequential_gap_filler(rasp.tokens, "-")
      gap_filler(["word1", None, "word3"])
      >> ["word1", "-", "word3"]
    """
    next_token = shift_by(-1, sop)
    gap_filler = rasp.SequenceMap(lambda x, y: filler if x is None and y is not None else x, sop, next_token)
    return gap_filler