from typing import Set

from circuits_benchmark.benchmark import vocabs
from circuits_benchmark.benchmark.program_evaluation_type import causal_and_regular
from circuits_benchmark.benchmark.tracr_benchmark_case import TracrBenchmarkCase
from tracr.rasp import rasp


class Case15(TracrBenchmarkCase):
  def get_program(self) -> rasp.SOp:
    return make_nary_sequencemap(lambda x, y, z: x + y - z, rasp.tokens, rasp.tokens, rasp.indices)

  def get_task_description(self) -> str:
    return "Returns each token multiplied by two and subtracted by its index."

  def get_vocab(self) -> Set:
    return vocabs.get_int_digits_vocab(count=5)

  def get_max_seq_len(self) -> int:
    return 5

@causal_and_regular
def make_nary_sequencemap(f, *sops):
  """Returns an SOp that simulates an n-ary SequenceMap.

  Uses multiple binary SequenceMaps to convert n SOps x_1, x_2, ..., x_n
  into a single SOp arguments that takes n-tuples as value. The n-ary sequence
  map implementing f is then a Map on this resulting SOp.

  Note that the intermediate variables representing tuples of varying length
  will be encoded categorically, and can become very high-dimensional. So,
  using this function might lead to very large compiled models.

  Args:
    f: Function with n arguments.
    *sops: Sequence of SOps, one for each argument of f.
  """
  values, *sops = sops
  for sop in sops:
    # x is a single entry in the first iteration but a tuple in later iterations
    values = rasp.SequenceMap(
        lambda x, y: (*x, y) if isinstance(x, tuple) else (x, y), values, sop)
  return rasp.Map(lambda args: f(*args), values)
