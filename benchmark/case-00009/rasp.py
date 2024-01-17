from typing import Set

from benchmark import vocabs
from tracr.rasp import rasp
from benchmark.program_evaluation_type import only_non_causal
from benchmark.common_programs import make_sort

@only_non_causal
def get_program() -> rasp.SOp:
  return make_sort(rasp.tokens, rasp.tokens, 10, 1)


def get_vocab() -> Set:
  return vocabs.get_int_digits_vocab()
