from tracr.rasp import rasp


def get_program() -> rasp.SOp:
  return make_secret_code_decoder(rasp.tokens)

def make_secret_code_decoder(sop: rasp.SOp) -> rasp.SOp:
    """
    Decodes a secret code by shifting each character a certain number of places in the alphabet.

    Example usage:
      code_decoder = make_secret_code_decoder(rasp.tokens)
      code_decoder(["uryyb", "jbeyq"], -13)  # Rot13 cipher
      >> ["hello", "world"]
    """
    def shift_char(c, shift):
        if c.isalpha():
            shifted = ord(c) + shift
            if c.islower():
                return chr((shifted - ord('a')) % 26 + ord('a'))
            else:
                return chr((shifted - ord('A')) % 26 + ord('A'))
        return c

    def decode(token, shift):
        return ''.join(shift_char(c, shift) for c in token)

    shift_value = rasp.Full(-13)  # Example: Rot13 cipher
    decoded_message = rasp.SequenceMap(decode, sop, shift_value)
    return decoded_message