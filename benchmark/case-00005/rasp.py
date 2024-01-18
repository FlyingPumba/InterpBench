from typing import Set

from benchmark.benchmark_case import BenchmarkCase
from benchmark.common_programs import make_shuffle_dyck
from tracr.rasp import rasp


class Case00005(BenchmarkCase):
  def get_program(self) -> rasp.SOp:
    return make_shuffle_dyck(pairs=["()", "{}"])

  def get_vocab(self) -> Set:
    return {"(", ")", "{", "}", "x"}