from typing import Set

from circuits_benchmark.benchmark.benchmark_case import BenchmarkCase
from circuits_benchmark.benchmark.common_programs import make_shuffle_dyck
from tracr.rasp import rasp


class Case5(BenchmarkCase):
  def get_program(self) -> rasp.SOp:
    return make_shuffle_dyck(pairs=["()", "{}"])

  def get_vocab(self) -> Set:
    return {"(", ")", "{", "}", "x"}

  def supports_causal_masking(self) -> bool:
    return False
