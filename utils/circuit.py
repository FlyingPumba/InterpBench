from networkx import DiGraph

from tracr.craft.transformers import SeriesWithResiduals
from utils.tracr_utils import get_relevant_component_names, get_output_space_basis_name


class Circuit(DiGraph):
  def add_node(self, transformer_component_name: str, label: str | None = None):
    super().add_node(transformer_component_name, label=label)

  @staticmethod
  def from_tracr(tracr_graph: DiGraph, craft_model: SeriesWithResiduals):
    component_names = get_relevant_component_names(craft_model)

    label_to_component_name_mapping = {}

    circuit = Circuit()

    for node_label, node_data in tracr_graph.nodes(data=True):
      component_name = None

      if node_label == "tokens":
        component_name = "embed.W_E"
      elif node_label == "indices":
        component_name = "pos_embed.W_pos"
      elif "MODEL_BLOCK" in node_data:
        node_block = node_data["MODEL_BLOCK"]
        output_basis_name = get_output_space_basis_name(node_block)
        matching_block_indices = [i for i, block in enumerate(craft_model.blocks) if
                                  get_output_space_basis_name(block) == output_basis_name]
        assert len(matching_block_indices) == 1, "Only one matching block is supported."
        block_index = matching_block_indices[0]
        component_name = component_names[block_index]

      if component_name is not None:
        label_to_component_name_mapping[node_label] = component_name
        circuit.add_node(component_name, label=node_label)

    for edge in tracr_graph.edges():
      from_label, to_label = edge
      if from_label in label_to_component_name_mapping and to_label in label_to_component_name_mapping:
        from_component_name = label_to_component_name_mapping[from_label]
        to_component_name = label_to_component_name_mapping[to_label]
        circuit.add_edge(from_component_name, to_component_name)

    return circuit

