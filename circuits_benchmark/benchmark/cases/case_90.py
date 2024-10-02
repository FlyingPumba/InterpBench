from typing import Set

from tracr.rasp import rasp

from circuits_benchmark.benchmark import vocabs
from circuits_benchmark.benchmark.tracr_benchmark_case import TracrBenchmarkCase


class Case90(TracrBenchmarkCase):
    def get_program(self) -> rasp.SOp:
        return make_token_replacer(rasp.tokens, "findme", "-")

    def get_task_description(self) -> str:
        return "Replaces a specific token with another one."

    def get_vocab(self) -> Set:
        vocab = vocabs.get_words_vocab()
        vocab.add("findme")
        vocab.add("-")
        return vocab

    def is_trivial(self) -> bool:
        return True


def make_token_replacer(sop: rasp.SOp, target: str, replacement: str) -> rasp.SOp:
    """
    Returns a program that replaces a target token with a replacement token

    Example usage:
      replacer = make_token_replacer(rasp.tokens, "findme", "-")
      replacer(["word1", "findme", "word3"])
      >> ["word1", "-", "word3"]
    """
    replaced = rasp.Map(lambda x: replacement if x == target else x, sop)
    return replaced
