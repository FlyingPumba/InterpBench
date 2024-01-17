from typing import Set

from benchmark import vocabs
from tracr.rasp import rasp
from benchmark.program_evaluation_type import causal_and_regular

@causal_and_regular
def get_program() -> rasp.SOp:
  return make_nary_sequencemap(lambda x, y, z: x + y - z, rasp.tokens, rasp.tokens, rasp.indices)

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


def get_vocab() -> Set:
  return vocabs.get_int_digits_vocab(count=5)


def get_max_seq_len() -> int:
  return 5
