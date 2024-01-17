from typing import Set

from benchmark import vocabs
from tracr.rasp import rasp
from benchmark.program_evaluation_type import only_non_causal
from benchmark.common_programs import make_hist

@only_non_causal
def get_program() -> rasp.SOp:
  return make_hist()


def get_vocab() -> Set:
  return vocabs.get_ascii_letters_vocab(count=3)