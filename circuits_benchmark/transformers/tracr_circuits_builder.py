from typing import Set

from networkx import DiGraph

from circuits_benchmark.transformers.alignment import Alignment
from circuits_benchmark.transformers.circuit import Circuit
from circuits_benchmark.transformers.circuit_granularity import CircuitGranularity
from tracr.compiler import nodes
from tracr.craft.bases import VectorSpaceWithBasis
from tracr.craft.transformers import SeriesWithResiduals, MultiAttentionHead, AttentionHead
from tracr.rasp.rasp import Aggregate, Selector


def build_tracr_circuits(tracr_graph: DiGraph,
                         craft_model: SeriesWithResiduals,
                         granularity: CircuitGranularity = "component") -> (Circuit, Circuit, Alignment):
  hl_circuit = build_hl_circuit(tracr_graph)
  ll_circuit = Circuit(granularity)
  alignment = Alignment()

  if "tokens" in hl_circuit.nodes:
    if granularity == "component":
      ll_node = "embed"
    else:
      ll_node = "embed.W_E"

    ll_circuit.add_node(ll_node)
    alignment.map_hl_to_ll("tokens", ll_node)

  if "indices" in hl_circuit.nodes:
    if granularity == "component":
      ll_node = "pos_embed"
    else:
      ll_node = "pos_embed.W_pos"

    ll_circuit.add_node(ll_node)
    alignment.map_hl_to_ll("indices", ll_node)

  layer = -1
  last_component_type = "mlp"
  for block in craft_model.blocks:
    if isinstance(block, MultiAttentionHead):
      assert len(block.sub_blocks) == 1, "Only one sub block is supported."
      block = block.sub_blocks[0]

    if isinstance(block, AttentionHead):
      # We always increase layer when we find an attention head
      layer += 1
      last_component_type = "attn"

      process_attention_head_block(block, layer, ll_circuit, granularity, alignment)
    else:
      if last_component_type == "mlp":
        # we only increase layer when we find an mlp and the last component was also an mlp
        layer += 1
      last_component_type = "mlp"

      process_mlp_block(block, layer, ll_circuit, granularity, alignment)

  # Assert that we have all the nodes and edges in the tracr_graph (which are also nodes in the hl_circuit).
  # Also, complete the alignment for the cases in which we have a selector or an aggregate expression.
  nodes_with_data = list(tracr_graph.nodes(data=True))
  for label, label_data in nodes_with_data:
    if nodes.MODEL_BLOCK in label_data:
      # This is label that corresponds directly to a block
      for node in alignment.get_ll_nodes(label):
        assert node in ll_circuit.nodes, f"Node {node} not found in circuit"
    else:
      # This is a label for which we don't have a component
      expr = label_data[nodes.EXPR]
      if isinstance(expr, Selector):
        # Selectors are computed in Q-K matrices. Let's find the Aggregate expression that uses this selector, and
        # then find the component implementing the Aggregate expression.
        for other_label, other_label_data in nodes_with_data:
          other_expr = other_label_data[nodes.EXPR]
          if isinstance(other_expr, Aggregate) and other_expr.selector == expr:
            ll_nodes = alignment.get_ll_nodes(other_label, remove_predecessors_by_ll_circuit=ll_circuit)
            assert len(ll_nodes) > 0, f"No nodes found for label {other_label}"
            for node in ll_nodes:
              if granularity == "component":
                # the same output node is the one that implements both the selector and the aggregate
                assert node in ll_circuit.nodes, f"Node {node} not found in circuit"
                alignment.map_hl_to_ll(label, node)
              elif granularity == "matrix":
                # the selector is implemented by the nodes that are input for the aggregate
                for pred_node in ll_circuit.predecessors(node):
                  if "W_Q" in pred_node or "W_K" in pred_node:
                    alignment.map_hl_to_ll(label, pred_node)
              else:
                raise ValueError(f"Granularity {granularity} not supported for Select expressions")
            break

  return hl_circuit, ll_circuit, alignment


def process_attention_head_block(block, layer, ll_circuit, granularity, alignment):
  if granularity == "component":
    add_node_and_edges_for_block(ll_circuit,
                                 f"blocks.{layer}.attn",
                                 alignment.get_ll_nodes(get_hl_labels_from_input_spaces([block.w_qk.left_space,
                                                                                         block.w_qk.right_space,
                                                                                         block.w_ov.input_space])))

    for label in get_hl_labels_from_input_spaces([block.w_ov.output_space]):
      alignment.map_hl_to_ll(label, f"blocks.{layer}.attn")

  elif granularity == "matrix":
    add_node_and_edges_for_block(ll_circuit,
                                 f"blocks.{layer}.attn.W_Q",
                                 alignment.get_ll_nodes(get_hl_labels_from_input_spaces([block.w_qk.left_space]),
                                                        remove_predecessors_by_ll_circuit=ll_circuit))
    add_node_and_edges_for_block(ll_circuit,
                                 f"blocks.{layer}.attn.W_K",
                                 alignment.get_ll_nodes(get_hl_labels_from_input_spaces([block.w_qk.right_space]),
                                                        remove_predecessors_by_ll_circuit=ll_circuit))

    add_node_and_edges_for_block(ll_circuit,
                                 f"blocks.{layer}.attn.W_V",
                                 alignment.get_ll_nodes(get_hl_labels_from_input_spaces([block.w_ov.input_space]),
                                                        remove_predecessors_by_ll_circuit=ll_circuit))
    for label in get_hl_labels_from_input_spaces([block.w_ov.output_space]):
      alignment.map_hl_to_ll(label, f"blocks.{layer}.attn.W_V")

    add_node_and_edges_for_block(ll_circuit,
                                 f"blocks.{layer}.attn.W_O",
                                 {f"blocks.{layer}.attn.W_Q", f"blocks.{layer}.attn.W_K", f"blocks.{layer}.attn.W_V"})
    for label in get_hl_labels_from_input_spaces([block.w_ov.output_space]):
      alignment.map_hl_to_ll(label, f"blocks.{layer}.attn.W_O")


def process_mlp_block(block, layer, ll_circuit, granularity, alignment):
  if granularity == "component":
    add_node_and_edges_for_block(ll_circuit,
                                 f"blocks.{layer}.mlp",
                                 alignment.get_ll_nodes(get_hl_labels_from_input_spaces([block.fst.input_space])))
    for label in get_hl_labels_from_input_spaces([block.snd.output_space]):
      alignment.map_hl_to_ll(label, f"blocks.{layer}.mlp")

  elif granularity == "matrix":
    add_node_and_edges_for_block(ll_circuit,
                                 f"blocks.{layer}.mlp.W_in",
                                 alignment.get_ll_nodes(get_hl_labels_from_input_spaces([block.fst.input_space]),
                                                        remove_predecessors_by_ll_circuit=ll_circuit))
    for label in get_hl_labels_from_input_spaces([block.snd.output_space]):
      alignment.map_hl_to_ll(label, f"blocks.{layer}.mlp.W_in")

    add_node_and_edges_for_block(ll_circuit,
                                 f"blocks.{layer}.mlp.W_out",
                                 {f"blocks.{layer}.mlp.W_in"})
    for label in get_hl_labels_from_input_spaces([block.snd.output_space]):
      alignment.map_hl_to_ll(label, f"blocks.{layer}.mlp.W_out")


def add_node_and_edges_for_block(ll_circuit: Circuit,
                                 node_name: str,
                                 input_ll_nodes: Set[str]):
  ll_circuit.add_node(node_name)
  for input_node in input_ll_nodes:
    ll_circuit.add_edge(input_node, node_name)


def get_hl_labels_from_input_spaces(input_spaces: list[VectorSpaceWithBasis]) -> list[str]:
  hl_labels = set()
  for space in input_spaces:
    hl_labels = hl_labels.union(get_labels_from_vector_space(space))
  return list(hl_labels)


def build_hl_circuit(tracr_graph: DiGraph) -> Circuit:
  hl_circuit = Circuit()

  for node in tracr_graph.nodes:
    hl_circuit.add_node(node)

  for edge in tracr_graph.edges:
    hl_circuit.add_edge(*edge)

  return hl_circuit


def get_labels_from_vector_space(space: VectorSpaceWithBasis):
  labels = set()
  for base in space.basis:
    if base.name == "one" and base.value is None:
      continue
    if base.name == "tokens" and base.value == "BOS":
      continue

    labels.add(base.name)
  return labels
