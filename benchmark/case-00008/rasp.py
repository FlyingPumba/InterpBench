from tracr.rasp import rasp
from benchmark.program_evaluation_type import only_non_causal
from benchmark.common_programs import make_sort_unique

@only_non_causal
def get_program() -> rasp.SOp:
  return make_sort_unique(rasp.tokens, rasp.tokens)