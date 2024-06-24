from dataclasses import dataclass
from typing import Set

from networkx import DiGraph

from circuits_benchmark.benchmark.vocabs import TRACR_BOS
from circuits_benchmark.utils.circuit.alignment import Alignment
from circuits_benchmark.utils.circuit.circuit import Circuit
from circuits_benchmark.utils.circuit.circuit_granularity import CircuitGranularity
from circuits_benchmark.utils.circuit.circuit_node import CircuitNode
from tracr.compiler import nodes
from tracr.craft.bases import VectorSpaceWithBasis
from tracr.craft.transformers import SeriesWithResiduals, MultiAttentionHead, AttentionHead, MLP
from tracr.rasp.rasp import Aggregate, Selector

@dataclass
class TracrCircuits:
  tracr_variables_circuit: Circuit
  tracr_transformer_circuit: Circuit
  alignment: Alignment


def build_tracr_circuits(tracr_graph: DiGraph,
                         craft_model: SeriesWithResiduals,
                         granularity: CircuitGranularity = "component") -> TracrCircuits:
  """Returns the high-level Tracr variables circuit, the low-level circuit for Tracr-generated transformers, and the
  alignment between both circuits."""
  tracr_variables_circuit = build_tracr_variables_circuit(tracr_graph)
  tracr_transformer_circuit = Circuit(granularity)
  alignment = Alignment()

  if "tokens" in tracr_variables_circuit.nodes:
    if granularity == "component":
      ll_node = "embed"
    elif granularity == "matrix":
      ll_node = "embed.W_E"
    else:
      ll_node = "hook_embed"

    tracr_transformer_circuit.add_node(CircuitNode(ll_node))
    alignment.map_hl_to_ll("tokens", CircuitNode(ll_node))

  if "indices" in tracr_variables_circuit.nodes:
    if granularity == "component":
      ll_node = "pos_embed"
    elif granularity == "matrix":
      ll_node = "pos_embed.W_pos"
    else:
      ll_node = "hook_pos_embed"

    tracr_transformer_circuit.add_node(CircuitNode(ll_node))
    alignment.map_hl_to_ll("indices", CircuitNode(ll_node))

  layer = -1
  last_component_type = "mlp"
  for block in craft_model.blocks:
    if isinstance(block, MultiAttentionHead):
      # We always increase layer when we find an attention head
      layer += 1
      last_component_type = "attn"

      for block_idx, block in enumerate(block.sub_blocks):
        if isinstance(block, AttentionHead):
          process_attention_head_block(block, block_idx, layer, tracr_transformer_circuit, granularity, alignment)
        else:
          raise ValueError(f"MultiAttentionHead sub-block {block_idx} is not an AttentionHead")

    if isinstance(block, MLP):
      if last_component_type == "mlp":
        # we only increase layer when we find an mlp and the last component was also an mlp
        layer += 1
      last_component_type = "mlp"

      process_mlp_block(block, layer, tracr_transformer_circuit, granularity, alignment)

  if granularity == "acdc_hooks" or granularity == "sp_hooks":
    resid_post_node = CircuitNode(f"blocks.{layer}.hook_resid_post")
    tracr_transformer_circuit.add_node(resid_post_node)

    hl_result_node = tracr_variables_circuit.get_result_node()
    for ll_node in alignment.get_ll_nodes(hl_result_node, remove_predecessors_by_ll_circuit=tracr_transformer_circuit):
      tracr_transformer_circuit.add_edge(ll_node, resid_post_node)
    alignment.map_hl_to_ll(hl_result_node, resid_post_node)

  # Assert that we have all the nodes and edges in the tracr_graph (which are also nodes in the tracr_variables_circuit).
  # Also, complete the alignment for the cases in which we have a selector or an aggregate expression.
  nodes_with_data = list(tracr_graph.nodes(data=True))
  for label, label_data in nodes_with_data:
    if nodes.MODEL_BLOCK in label_data:
      # This is label that corresponds directly to a block
      for ll_node in alignment.get_ll_nodes(label):
        assert ll_node in tracr_transformer_circuit.nodes, f"Node {ll_node} not found in circuit"
    else:
      # This is a label for which we don't have a component
      expr = label_data[nodes.EXPR]
      if isinstance(expr, Selector):
        # Selectors are computed in Q-K matrices. Let's find the Aggregate expression that uses this selector, and
        # then find the component implementing the Aggregate expression.
        for other_label, other_label_data in nodes_with_data:
          other_expr = other_label_data[nodes.EXPR]
          if isinstance(other_expr, Aggregate) and other_expr.selector == expr:
            ll_nodes = alignment.get_ll_nodes(other_label, remove_predecessors_by_ll_circuit=tracr_transformer_circuit)
            assert len(ll_nodes) > 0, f"No nodes found for label {other_label}"
            for ll_node in ll_nodes:
              if granularity == "component":
                # the same output node is the one that implements both the selector and the aggregate
                assert ll_node in tracr_transformer_circuit.nodes, f"Node {ll_node} not found in circuit"
                alignment.map_hl_to_ll(label, ll_node)
              elif granularity == "matrix":
                # the selector is implemented by the nodes that are input for the aggregate
                for pred_node in tracr_transformer_circuit.predecessors(ll_node):
                  if "W_Q" in pred_node.name or "W_K" in pred_node.name:
                    alignment.map_hl_to_ll(label, pred_node)
              elif granularity == "acdc_hooks" or granularity == "sp_hooks":
                # the selector is implemented by the nodes that are input for the aggregate
                for pred_node in tracr_transformer_circuit.predecessors(ll_node):
                  if "hook_q" in pred_node.name or "hook_k" in pred_node.name:
                    alignment.map_hl_to_ll(label, pred_node)
              else:
                raise ValueError(f"Granularity {granularity} not supported for Select expressions")
            break

  return TracrCircuits(tracr_variables_circuit, tracr_transformer_circuit, alignment)


def process_attention_head_block(block: AttentionHead,
                                 head_index: int,
                                 layer: int,
                                 ll_circuit: Circuit,
                                 granularity: CircuitGranularity,
                                 alignment: Alignment):
  if granularity == "component":
    add_node_and_edges_for_block(ll_circuit,
                                 CircuitNode(f"blocks.{layer}.attn", head_index),
                                 alignment.get_ll_nodes(get_hl_labels_from_input_spaces([block.w_qk.left_space,
                                                                                         block.w_qk.right_space,
                                                                                         block.w_ov.input_space])))

    for label in get_hl_labels_from_input_spaces([block.w_ov.output_space]):
      alignment.map_hl_to_ll(label, CircuitNode(f"blocks.{layer}.attn", head_index))

  elif granularity == "matrix":
    add_node_and_edges_for_block(ll_circuit,
                                 CircuitNode(f"blocks.{layer}.attn.W_Q", head_index),
                                 alignment.get_ll_nodes(get_hl_labels_from_input_spaces([block.w_qk.left_space]),
                                                        remove_predecessors_by_ll_circuit=ll_circuit))
    add_node_and_edges_for_block(ll_circuit,
                                 CircuitNode(f"blocks.{layer}.attn.W_K", head_index),
                                 alignment.get_ll_nodes(get_hl_labels_from_input_spaces([block.w_qk.right_space]),
                                                        remove_predecessors_by_ll_circuit=ll_circuit))

    add_node_and_edges_for_block(ll_circuit,
                                 CircuitNode(f"blocks.{layer}.attn.W_V", head_index),
                                 alignment.get_ll_nodes(get_hl_labels_from_input_spaces([block.w_ov.input_space]),
                                                        remove_predecessors_by_ll_circuit=ll_circuit))
    for label in get_hl_labels_from_input_spaces([block.w_ov.output_space]):
      alignment.map_hl_to_ll(label, CircuitNode(f"blocks.{layer}.attn.W_V", head_index))

    add_node_and_edges_for_block(ll_circuit,
                                 CircuitNode(f"blocks.{layer}.attn.W_O", head_index),
                                 {
                                   CircuitNode(f"blocks.{layer}.attn.W_Q", head_index),
                                   CircuitNode(f"blocks.{layer}.attn.W_K", head_index),
                                   CircuitNode(f"blocks.{layer}.attn.W_V", head_index)
                                 })
    for label in get_hl_labels_from_input_spaces([block.w_ov.output_space]):
      alignment.map_hl_to_ll(label, CircuitNode(f"blocks.{layer}.attn.W_O", head_index))

  elif granularity == "acdc_hooks" or granularity == "sp_hooks":
    add_node_and_edges_for_block(ll_circuit,
                                 CircuitNode(f"blocks.{layer}.hook_q_input", head_index),
                                 alignment.get_ll_nodes(get_hl_labels_from_input_spaces([block.w_qk.left_space]),
                                                        remove_predecessors_by_ll_circuit=ll_circuit))
    add_node_and_edges_for_block(ll_circuit,
                                 CircuitNode(f"blocks.{layer}.hook_k_input", head_index),
                                 alignment.get_ll_nodes(get_hl_labels_from_input_spaces([block.w_qk.right_space]),
                                                        remove_predecessors_by_ll_circuit=ll_circuit))

    add_node_and_edges_for_block(ll_circuit,
                                 CircuitNode(f"blocks.{layer}.hook_v_input", head_index),
                                 alignment.get_ll_nodes(get_hl_labels_from_input_spaces([block.w_ov.input_space]),
                                                        remove_predecessors_by_ll_circuit=ll_circuit))
    for label in get_hl_labels_from_input_spaces([block.w_ov.output_space]):
      alignment.map_hl_to_ll(label, CircuitNode(f"blocks.{layer}.hook_v_input", head_index))

    if granularity == "acdc_hooks":
      add_node_and_edges_for_block(ll_circuit,
                                   CircuitNode(f"blocks.{layer}.attn.hook_q", head_index),
                                   {CircuitNode(f"blocks.{layer}.hook_q_input", head_index)})
      for label in get_hl_labels_from_input_spaces([block.w_qk.left_space]):
        alignment.map_hl_to_ll(label, CircuitNode(f"blocks.{layer}.attn.hook_q", head_index))

      add_node_and_edges_for_block(ll_circuit,
                                   CircuitNode(f"blocks.{layer}.attn.hook_k", head_index),
                                   {CircuitNode(f"blocks.{layer}.hook_k_input", head_index)})
      for label in get_hl_labels_from_input_spaces([block.w_qk.right_space]):
        alignment.map_hl_to_ll(label, CircuitNode(f"blocks.{layer}.attn.hook_k", head_index))

      add_node_and_edges_for_block(ll_circuit,
                                   CircuitNode(f"blocks.{layer}.attn.hook_v", head_index),
                                   {CircuitNode(f"blocks.{layer}.hook_v_input", head_index)})
      for label in get_hl_labels_from_input_spaces([block.w_ov.output_space]):
        alignment.map_hl_to_ll(label, CircuitNode(f"blocks.{layer}.attn.hook_v", head_index))

      attn_result_input_edges = {CircuitNode(f"blocks.{layer}.attn.hook_q", head_index),
                                 CircuitNode(f"blocks.{layer}.attn.hook_k", head_index),
                                 CircuitNode(f"blocks.{layer}.attn.hook_v", head_index)}
    else:
      attn_result_input_edges = {CircuitNode(f"blocks.{layer}.hook_q_input", head_index),
                                 CircuitNode(f"blocks.{layer}.hook_k_input", head_index),
                                 CircuitNode(f"blocks.{layer}.hook_v_input", head_index)}

    add_node_and_edges_for_block(ll_circuit,
                                 CircuitNode(f"blocks.{layer}.attn.hook_result", head_index),
                                 attn_result_input_edges)
    for label in get_hl_labels_from_input_spaces([block.w_ov.output_space]):
      alignment.map_hl_to_ll(label, CircuitNode(f"blocks.{layer}.attn.hook_result", head_index))


def process_mlp_block(block: MLP,
                      layer: int,
                      ll_circuit: Circuit,
                      granularity: CircuitGranularity,
                      alignment: Alignment):
  if granularity == "component":
    add_node_and_edges_for_block(ll_circuit,
                                 CircuitNode(f"blocks.{layer}.mlp"),
                                 alignment.get_ll_nodes(get_hl_labels_from_input_spaces([block.fst.input_space])))
    for label in get_hl_labels_from_input_spaces([block.snd.output_space]):
      alignment.map_hl_to_ll(label, CircuitNode(f"blocks.{layer}.mlp"))

  elif granularity == "matrix":
    add_node_and_edges_for_block(ll_circuit,
                                 CircuitNode(f"blocks.{layer}.mlp.W_in"),
                                 alignment.get_ll_nodes(get_hl_labels_from_input_spaces([block.fst.input_space]),
                                                        remove_predecessors_by_ll_circuit=ll_circuit))
    for label in get_hl_labels_from_input_spaces([block.snd.output_space]):
      alignment.map_hl_to_ll(label, CircuitNode(f"blocks.{layer}.mlp.W_in"))

    add_node_and_edges_for_block(ll_circuit,
                                 CircuitNode(f"blocks.{layer}.mlp.W_out"),
                                 {CircuitNode(f"blocks.{layer}.mlp.W_in")})
    for label in get_hl_labels_from_input_spaces([block.snd.output_space]):
      alignment.map_hl_to_ll(label, CircuitNode(f"blocks.{layer}.mlp.W_out"))

  else:
    add_node_and_edges_for_block(ll_circuit,
                                 CircuitNode(f"blocks.{layer}.hook_mlp_in"),
                                 alignment.get_ll_nodes(get_hl_labels_from_input_spaces([block.fst.input_space]),
                                                        remove_predecessors_by_ll_circuit=ll_circuit))
    for label in get_hl_labels_from_input_spaces([block.snd.output_space]):
      alignment.map_hl_to_ll(label, CircuitNode(f"blocks.{layer}.hook_mlp_in"))

    add_node_and_edges_for_block(ll_circuit,
                                 CircuitNode(f"blocks.{layer}.hook_mlp_out"),
                                 {CircuitNode(f"blocks.{layer}.hook_mlp_in")})
    for label in get_hl_labels_from_input_spaces([block.snd.output_space]):
      alignment.map_hl_to_ll(label, CircuitNode(f"blocks.{layer}.hook_mlp_out"))


def add_node_and_edges_for_block(ll_circuit: Circuit,
                                 node: CircuitNode,
                                 input_ll_nodes: Set[CircuitNode]):
  ll_circuit.add_node(node)
  for input_node in input_ll_nodes:
    ll_circuit.add_edge(input_node, node)


def get_hl_labels_from_input_spaces(input_spaces: list[VectorSpaceWithBasis]) -> list[str]:
  hl_labels = set()
  for space in input_spaces:
    hl_labels = hl_labels.union(get_labels_from_vector_space(space))
  return list(hl_labels)


def build_tracr_variables_circuit(tracr_graph: DiGraph) -> Circuit:
  hl_circuit = Circuit()

  for node in tracr_graph.nodes:
    hl_circuit.add_node(CircuitNode(node))

  for from_node, to_node in tracr_graph.edges:
    hl_circuit.add_edge(CircuitNode(from_node), CircuitNode(to_node))

  return hl_circuit


def get_labels_from_vector_space(space: VectorSpaceWithBasis):
  labels = set()
  for base in space.basis:
    if base.name == "one" and base.value is None:
      continue
    if base.name == "tokens" and base.value == TRACR_BOS:
      continue

    labels.add(base.name)
  return labels
