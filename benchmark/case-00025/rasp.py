from tracr.rasp import rasp
from benchmark.common_programs import make_hist, make_length


def get_program() -> rasp.SOp:
  return make_token_frequency_normalization(rasp.tokens)

def make_token_frequency_normalization(sop: rasp.SOp) -> rasp.SOp:
    """
    Normalizes token frequencies in a sequence to a range between 0 and 1.

    Example usage:
      token_freq_norm = make_token_frequency_normalization(rasp.tokens)
      token_freq_norm(["a", "a", "b", "c", "c", "c"])
      >> [0.33, 0.33, 0.16, 0.5, 0.5, 0.5]
    """
    hist = make_hist()
    normalized_freq = rasp.Map(lambda x: x / make_length(), hist)
    return normalized_freq