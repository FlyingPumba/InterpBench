from tracr.rasp import rasp
from benchmark.program_evaluation_type import causal_and_regular
from benchmark.common_programs import make_frac_prevs

@causal_and_regular
def get_program() -> rasp.SOp:
  return make_frac_prevs(rasp.tokens)