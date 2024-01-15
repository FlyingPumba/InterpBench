from tracr.rasp import rasp
from benchmark.common_programs import make_hist, make_length


def get_program() -> rasp.SOp:
  return make_token_frequency_deviation(rasp.tokens)

def make_token_frequency_deviation(sop: rasp.SOp) -> rasp.SOp:
    """
    Calculates the deviation of each token's frequency from the average frequency in the sequence.

    Example usage:
      frequency_deviation = make_token_frequency_deviation(rasp.tokens)
      frequency_deviation(["a", "b", "a", "c", "a", "b"])
      >> [0.33, -0.33, 0.33, -0.66, 0.33, -0.33]
    """
    hist = make_hist()
    average_freq = rasp.Aggregate(
        rasp.Select(rasp.indices, rasp.indices, rasp.Comparison.TRUE),
        rasp.numerical(hist), default=0) / make_length()
    freq_deviation = rasp.Map(lambda x: x - average_freq, hist)
    return freq_deviation