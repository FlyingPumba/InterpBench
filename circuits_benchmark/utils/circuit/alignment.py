from typing import Dict, Set, List

from circuits_benchmark.utils.circuit.circuit import Circuit
from circuits_benchmark.utils.circuit.circuit_node import CircuitNode


class Alignment(object):
  def __init__(self):
    self.hl_to_ll_mapping: Dict[str, Set[CircuitNode]] = {}

  def map_hl_to_ll(self, hl_node: str, ll_node: CircuitNode):
    if hl_node not in self.hl_to_ll_mapping:
      self.hl_to_ll_mapping[hl_node] = set()
    self.hl_to_ll_mapping[hl_node].add(ll_node)

  def get_ll_nodes(self, hl_node: str | List[str] | CircuitNode,
                   remove_predecessors_by_ll_circuit: Circuit | None = None,
                   remove_successors_by_ll_circuit: Circuit | None = None) -> Set[CircuitNode]:
    hl_nodes = set()

    if isinstance(hl_node, str):
      hl_nodes.add(hl_node)
    elif isinstance(hl_node, CircuitNode):
      hl_nodes.add(hl_node.name)
    else:
      hl_nodes = set(hl_node)

    ll_nodes = set()
    for hl_node in hl_nodes:
      ll_nodes = ll_nodes.union(self.hl_to_ll_mapping[hl_node])

    if remove_predecessors_by_ll_circuit is not None:
      # remove all nodes in ll_nodes that are predecessors of other nodes in ll_nodes (even if it is a predecessor of a
      # predecessor, and so on)
      nodes_to_remove = set()
      nodes_to_check = ll_nodes.copy()
      while len(nodes_to_check) > 0:
        node = nodes_to_check.pop()
        for pred_node in remove_predecessors_by_ll_circuit.predecessors(node):
          if pred_node in ll_nodes:
            nodes_to_remove.add(pred_node)
            nodes_to_check.add(pred_node)

      ll_nodes = ll_nodes.difference(nodes_to_remove)

    if remove_successors_by_ll_circuit is not None:
      # remove all nodes in ll_nodes that are successors of other nodes in ll_nodes (even if it is a successor of a
      # successor, and so on)
      nodes_to_remove = set()
      nodes_to_check = ll_nodes.copy()
      while len(nodes_to_check) > 0:
        node = nodes_to_check.pop()
        for succ_node in remove_successors_by_ll_circuit.successors(node):
          if succ_node in ll_nodes:
            nodes_to_remove.add(succ_node)
            nodes_to_check.add(succ_node)

      ll_nodes = ll_nodes.difference(nodes_to_remove)

    return ll_nodes
