from acdc.TLACDCCorrespondence import TLACDCCorrespondence
from circuits_benchmark.transformers.circuit import Circuit
from circuits_benchmark.transformers.circuit_node import CircuitNode


def build_acdc_circuit(corr: TLACDCCorrespondence) -> Circuit:
  circuit = Circuit()

  for (child_name, child_index, parent_name, parent_index), edge in corr.edge_dict().items():
    if edge.present:
      parent_head_index = None
      if (parent_index is not None and
          len(parent_index.hashable_tuple) > 2 and
          parent_index.hashable_tuple[2] is not None):
        parent_head_index = parent_index.hashable_tuple[2]

      child_head_index = None
      if (child_index is not None and
          len(child_index.hashable_tuple) > 2 and
          child_index.hashable_tuple[2] is not None):
        child_head_index = child_index.hashable_tuple[2]

      from_node = CircuitNode(parent_name, parent_head_index)
      to_node = CircuitNode(child_name, child_head_index)
      circuit.add_edge(from_node, to_node)

  return circuit


def get_full_acdc_circuit(n_layers: int, n_heads: int) -> Circuit:
  circuit = Circuit()

  circuit.add_node(CircuitNode("hook_embed"))
  circuit.add_node(CircuitNode("hook_pos_embed"))

  resid_writers = ["hook_embed", "hook_pos_embed", "attn.hook_result", "mlp.hook_out"]

  for layer in range(n_layers):
    # attention heads
    for head in range(n_heads):
      for letter in "qkv":
        input_node = CircuitNode(f"blocks.{layer}.hook_{letter}_input", head)
        matrix_node = CircuitNode(f"blocks.{layer}.attn.hook_{letter}", head)
        circuit.add_node(input_node)
        circuit.add_node(matrix_node)
        circuit.add_edge(input_node, matrix_node)

      attn_result_node = CircuitNode(f"blocks.{layer}.attn.hook_result", head)
      circuit.add_node(attn_result_node)
      for letter in "qkv":
        matrix_node = CircuitNode(f"blocks.{layer}.attn.hook_{letter}", head)
        circuit.add_edge(matrix_node, attn_result_node)

      current_nodes = list(circuit.nodes)
      nodes_that_receive_resid_directly = [f"blocks.{layer}.hook_q_input",
                                           f"blocks.{layer}.hook_k_input",
                                           f"blocks.{layer}.hook_v_input"]
      for from_node in current_nodes:
        if any([from_node.name.endswith(resid_writer) for resid_writer in resid_writers]):
          if f"blocks.{layer}" not in from_node.name:
            for to_node_name in nodes_that_receive_resid_directly:
              circuit.add_edge(from_node, CircuitNode(to_node_name, head))

    # mlp
    mlp_in_node = CircuitNode(f"blocks.{layer}.hook_mlp_in")
    mlp_out_node = CircuitNode(f"blocks.{layer}.hook_mlp_out")
    circuit.add_node(mlp_in_node)
    circuit.add_node(mlp_out_node)
    circuit.add_edge(mlp_in_node, mlp_out_node)

    current_nodes = list(circuit.nodes)
    nodes_that_receive_resid_directly = [mlp_in_node]
    for from_node in current_nodes:
      if any([from_node.name.endswith(resid_writer) for resid_writer in resid_writers]):
        if (f"blocks.{layer}" not in from_node.name) or ("attn.hook_result" in from_node.name):
          for to_node in nodes_that_receive_resid_directly:
            circuit.add_edge(from_node, to_node)

  last_resid_post_node = CircuitNode(f"blocks.{n_layers - 1}.hook_resid_post")
  circuit.add_node(last_resid_post_node)
  for from_node in list(circuit.nodes):
    if any([from_node.name.endswith(resid_writer) for resid_writer in resid_writers]):
      circuit.add_edge(from_node, last_resid_post_node)

  return circuit