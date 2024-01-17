from typing import Set

from benchmark import vocabs
from tracr.rasp import rasp
from benchmark.program_evaluation_type import causal_and_regular
from benchmark.common_programs import make_frac_prevs


@causal_and_regular
def get_program() -> rasp.SOp:
  return make_frac_prevs(rasp.tokens == "x")

def get_vocab() -> Set:
  some_letters = vocabs.get_ascii_letters_vocab(count=3)
  some_letters.add("x")
  return some_letters