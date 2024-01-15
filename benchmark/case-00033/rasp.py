from tracr.rasp import rasp


def get_program() -> rasp.SOp:
  return make_token_length_parity_checker(rasp.tokens)

def make_token_length_parity_checker(sop: rasp.SOp) -> rasp.SOp:
    """
    Checks if each token's length is odd or even.

    Example usage:
      length_parity = make_token_length_parity_checker(rasp.tokens)
      length_parity(["hello", "worlds", "!", "2022"])
      >> [False, True, False, True]
    """
    length_parity_checker = rasp.Map(lambda x: len(x) % 2 == 0, sop)
    return length_parity_checker