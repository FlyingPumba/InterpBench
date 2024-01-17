from typing import Set

from benchmark import vocabs
from tracr.rasp import rasp
from benchmark.program_evaluation_type import causal_and_regular
from benchmark.common_programs import detect_pattern


@causal_and_regular
def get_program() -> rasp.SOp:
  return detect_pattern(rasp.tokens, "abc")


def get_vocab() -> Set:
  return vocabs.get_ascii_letters_vocab(count=3)


def get_max_seq_len() -> int:
  return 15
