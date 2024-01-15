from tracr.rasp import rasp
from benchmark.common_programs import shift_by


def get_program() -> rasp.SOp:
  return make_sequential_token_distance_measurement(rasp.tokens)

def make_sequential_token_distance_measurement(sop: rasp.SOp) -> rasp.SOp:
    """
    Measures the distance between sequential tokens in terms of the number of tokens in between.

    Example usage:
      token_distance = make_sequential_token_distance_measurement(rasp.tokens)
      token_distance(["a", "b", "c", "a", "d"])
      >> [3, 3, 3, 0, 3]
    """
    prev_indices = shift_by(1, rasp.indices)
    token_distance = rasp.SequenceMap(lambda x, y: abs(x - y) if None not in [x, y] else None, rasp.indices, prev_indices)
    return token_distance