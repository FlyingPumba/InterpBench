from tracr.rasp import rasp
from benchmark.common_programs import shift_by


def get_program() -> rasp.SOp:
  return make_sequential_duplicate_removal(rasp.tokens)

def make_sequential_duplicate_removal(sop: rasp.SOp) -> rasp.SOp:
    """
    Removes consecutive duplicate tokens from a sequence.

    Example usage:
      duplicate_remove = make_sequential_duplicate_removal(rasp.tokens)
      duplicate_remove("aabbcc")
      >> ['a', None, 'b', None, 'c', None]

    Args:
      sop: SOp representing the sequence to process.

    Returns:
      A SOp that maps an input sequence to another sequence where immediate 
      duplicate occurrences of any token are removed.
    """
    shifted_sop = shift_by(1, sop)
    duplicate_removal_sop = rasp.SequenceMap(
        lambda x, y: x if x != y else None, sop, shifted_sop).named("sequential_duplicate_removal")
    return duplicate_removal_sop