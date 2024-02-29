from typing import Dict, Set, List

from circuits_benchmark.transformers.circuit import Circuit


class Alignment(object):
  def __init__(self):
    self.hl_to_ll_mapping: Dict[str, Set[str]] = {}

  def map_hl_to_ll(self, hl_node: str, ll_node: str):
    if hl_node not in self.hl_to_ll_mapping:
      self.hl_to_ll_mapping[hl_node] = set()
    self.hl_to_ll_mapping[hl_node].add(ll_node)

  def get_ll_nodes(self, hl_node: str | List[str],
                   remove_predecessors_by_ll_circuit: Circuit | None = None,
                   remove_successors_by_ll_circuit: Circuit | None = None) -> Set[str]:
    hl_nodes = set()

    if isinstance(hl_node, str):
      hl_nodes.add(hl_node)
    else:
      hl_nodes = set(hl_node)

    ll_nodes = set()
    for node in hl_nodes:
      ll_nodes = ll_nodes.union(self.hl_to_ll_mapping[node])

    if remove_predecessors_by_ll_circuit is not None:
      # remove all nodes in ll_nodes that are predecessors of other nodes in ll_nodes
      nodes_to_remove = set()
      for node in ll_nodes:
        for pred_node in remove_predecessors_by_ll_circuit.predecessors(node):
          if pred_node in ll_nodes:
            nodes_to_remove.add(pred_node)

      ll_nodes = ll_nodes.difference(nodes_to_remove)

    if remove_successors_by_ll_circuit is not None:
      # remove all nodes in ll_nodes that are successors of other nodes in ll_nodes
      nodes_to_remove = set()
      for node in ll_nodes:
        for succ_node in remove_successors_by_ll_circuit.successors(node):
          if succ_node in ll_nodes:
            nodes_to_remove.add(succ_node)

      ll_nodes = ll_nodes.difference(nodes_to_remove)

    return ll_nodes
