from typing import List

from networkx import DiGraph

from tracr.craft.transformers import SeriesWithResiduals
from circuits_benchmark.utils.tracr_utils import get_relevant_component_names, get_output_space_basis_name
from tracr.compiler import nodes
from tracr.rasp.rasp import Aggregate


class Circuit(DiGraph):
  def add_node(self, transformer_component_name: str, labels: List[str] | None = None):
    super().add_node(transformer_component_name, labels=labels)

  @staticmethod
  def from_tracr(tracr_graph: DiGraph, craft_model: SeriesWithResiduals):
    circuit = Circuit()

    component_names = get_relevant_component_names(craft_model)
    label_to_component_name_mapping = {}

    # Add nodes to the circuit
    for node_label, node_data in tracr_graph.nodes(data=True):
      component_name = None
      labels = [node_label]

      if node_label == "tokens":
        component_name = "embed.W_E"
      elif node_label == "indices":
        component_name = "pos_embed.W_pos"
      elif nodes.MODEL_BLOCK in node_data and nodes.EXPR in node_data:
        node_block = node_data[nodes.MODEL_BLOCK]
        output_basis_name = get_output_space_basis_name(node_block)
        matching_block_indices = [i for i, block in enumerate(craft_model.blocks) if
                                  get_output_space_basis_name(block) == output_basis_name]
        assert len(matching_block_indices) == 1, "Only one matching block is supported."
        block_index = matching_block_indices[0]
        component_name = component_names[block_index]

        node_expr = node_data[nodes.EXPR]
        if isinstance(node_expr, Aggregate):
          # Aggregators use Selectors, which are implemented as part of the same component, so we need to map two labels
          # to the same component name.
          labels.append(node_expr.selector.label)

      if component_name is not None:
        for label in labels:
          label_to_component_name_mapping[label] = component_name
        circuit.add_node(component_name, labels=labels)

    # assert all nodes have been assigned as labels in the circuit
    assert len(label_to_component_name_mapping.keys()) == len(tracr_graph.nodes)

    # Add edges to the circuit
    for edge in tracr_graph.edges():
      from_label, to_label = edge
      from_component_name = label_to_component_name_mapping[from_label]
      to_component_name = label_to_component_name_mapping[to_label]

      if from_component_name != to_component_name:
        circuit.add_edge(from_component_name, to_component_name)

    return circuit

