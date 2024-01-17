from typing import Set

from benchmark import vocabs
from tracr.rasp import rasp
from benchmark.program_evaluation_type import causal_and_regular
from benchmark.common_programs import shift_by

@causal_and_regular
def get_program() -> rasp.SOp:
  return shift_by(2, rasp.tokens)


def get_vocab() -> Set:
  return vocabs.get_str_digits_vocab()