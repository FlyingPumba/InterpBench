from tracr.rasp import rasp
from benchmark.program_evaluation_type import only_non_causal
from benchmark.common_programs import make_hist, make_sort
from benchmark.defaults import default_max_seq_len


@only_non_causal
def get_program() -> rasp.SOp:
  return make_sort_freq(default_max_seq_len)

@only_non_causal
def make_sort_freq(max_seq_len: int) -> rasp.SOp:
  """Returns tokens sorted by the frequency they appear in the input.

  Tokens the appear the same amount of times are output in the same order as in
  the input.

  Example usage:
    sort = make_sort_freq(rasp.tokens, rasp.tokens, 5)
    sort([2, 4, 2, 1])
    >> [2, 2, 4, 1]

  Args:
    max_seq_len: Maximum sequence length (used to ensure keys are unique)
  """
  hist = -1 * make_hist().named("hist")
  return make_sort(
      rasp.tokens, hist, max_seq_len=max_seq_len, min_key=1).named("sort_freq")