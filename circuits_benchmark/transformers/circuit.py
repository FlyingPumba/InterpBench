from networkx import DiGraph

from circuits_benchmark.transformers.circuit_granularity import CircuitGranularity


class Circuit(DiGraph):
  def __init__(self, granularity: CircuitGranularity | None = None, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.granularity = granularity
