from typing import Set

from benchmark import vocabs
from benchmark.common_programs import make_length
from benchmark.program_evaluation_type import only_non_causal
from tracr.rasp import rasp


@only_non_causal
def get_program() -> rasp.SOp:
    return make_length()


def get_vocab() -> Set:
    return vocabs.get_ascii_letters_vocab()