from typing import Set

from benchmark import vocabs
from tracr.rasp import rasp
from benchmark.program_evaluation_type import only_non_causal
from benchmark.common_programs import make_pair_balance

@only_non_causal
def get_program() -> rasp.SOp:
  return make_pair_balance(rasp.tokens, "(", ")")


def get_vocab() -> Set:
  some_letters = vocabs.get_ascii_letters_vocab(count=3)
  some_letters.add("x")
  return some_letters