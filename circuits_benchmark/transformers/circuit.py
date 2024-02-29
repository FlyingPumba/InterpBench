from __future__ import annotations

from networkx import DiGraph

from circuits_benchmark.transformers.circuit_granularity import CircuitGranularity
from circuits_benchmark.utils.cloudpickle import dump_to_pickle, load_from_pickle


class Circuit(DiGraph):
  def __init__(self, granularity: CircuitGranularity | None = None, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.granularity = granularity

  def save(self, file_path: str):
    if not file_path.endswith(".pkl"):
      file_path += ".pkl"

    dump_to_pickle(file_path, self)

  @staticmethod
  def load(file_path) -> Circuit | None:
    return load_from_pickle(file_path)

  def get_result_node(self):
    """Returns the node in the circuit that doesn't have successors (there should be only one)."""
    result_nodes = [node for node in self.nodes if not list(self.successors(node))]
    assert len(result_nodes) == 1, f"Expected 1 result node, got {len(result_nodes)}"
    return result_nodes[0]
