from __future__ import annotations

from functools import cached_property

from networkx import DiGraph

from circuits_benchmark.transformers.circuit_granularity import CircuitGranularity
from circuits_benchmark.transformers.circuit_node import CircuitNode
from circuits_benchmark.transformers.circuit_node_view import CircuitNodeView
from circuits_benchmark.utils.cloudpickle import dump_to_pickle, load_from_pickle


class Circuit(DiGraph):
  def __init__(self, granularity: CircuitGranularity | None = None, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.granularity = granularity

  def add_node(self, node_for_adding: CircuitNode, **attr):
    # Make sure that node_for_adding is a CircuitNode
    if not isinstance(node_for_adding, CircuitNode):
      raise ValueError(f"Expected a CircuitNode, got {type(node_for_adding)}")

    super().add_node(node_for_adding, **attr)

  def add_edge(self, u_of_edge: CircuitNode, v_of_edge: CircuitNode, **attr):
    # Make sure that u_of_edge and v_of_edge are CircuitNodes
    if not isinstance(u_of_edge, CircuitNode) or not isinstance(v_of_edge, CircuitNode):
      raise ValueError(f"Expected a CircuitNode, got {type(u_of_edge)} and {type(v_of_edge)}")

    super().add_edge(u_of_edge, v_of_edge, **attr)

  @cached_property
  def nodes(self):
    return CircuitNodeView(self)

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
