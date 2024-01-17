from typing import Set

from benchmark import vocabs
from tracr.rasp import rasp
from benchmark.common_programs import make_unique_token_extractor


def get_program() -> rasp.SOp:
  return make_unique_token_extractor(rasp.tokens)


def get_vocab() -> Set:
  return vocabs.get_ascii_letters_vocab(count=3)