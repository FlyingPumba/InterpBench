from tracr.rasp import rasp
from benchmark.common_programs import make_unique_token_extractor


def get_program() -> rasp.SOp:
  return make_unique_token_extractor(rasp.tokens)