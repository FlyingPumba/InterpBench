from tracr.rasp import rasp
from benchmark.common_programs import detect_pattern


def get_program() -> rasp.SOp:
  return make_nested_pattern_extraction(rasp.tokens, "(", ")")

def make_nested_pattern_extraction(sop: rasp.SOp, open_token: str, close_token: str) -> rasp.SOp:
    """
    Extracts nested patterns like parentheses or HTML tags from a sequence.

    Example usage:
      nested_pattern = make_nested_pattern_extraction(rasp.tokens, "(", ")")
      nested_pattern("(a(b)c)(d)")
      >> [((True, False), (False, False)), ...]

    Args:
      sop: SOp representing the sequence to analyze.
      open_token: The token representing the start of a nested pattern.
      close_token: The token representing the end of a nested pattern.

    Returns:
      A SOp that maps an input sequence to a sequence of tuples, each indicating 
      the start and end of a nested pattern.
    """
    open_detector = detect_pattern(sop, [open_token])
    close_detector = detect_pattern(sop, [close_token])
    nested_pattern_sop = rasp.SequenceMap(
        lambda x, y: (x, y), open_detector, close_detector).named("nested_pattern_extraction")
    return nested_pattern_sop