from acdc.TLACDCCorrespondence import TLACDCCorrespondence
from circuits_benchmark.transformers.circuit import Circuit


def build_acdc_circuit(corr: TLACDCCorrespondence) -> Circuit:
  circuit = Circuit()

  for (child_name, child_index, parent_name, parent_index), edge in corr.all_edges().items():
    if edge.present:
      from_node = f"{parent_name}{str(parent_index)}"
      to_node = f"{child_name}{str(child_index)}"
      circuit.add_edge(from_node, to_node)

  return circuit


def get_full_acdc_circuit(n_layers: int) -> Circuit:
  circuit = Circuit()

  circuit.add_node("hook_embed")
  circuit.add_node("hook_pos_embed")

  resid_writers = ["hook_embed", "hook_pos_embed", "attn.hook_result", "mlp.hook_out"]

  for layer in range(n_layers):
    # attention head
    for letter in "qkv":
      circuit.add_node(f"blocks.{layer}.hook_{letter}_input")
      circuit.add_node(f"blocks.{layer}.attn.hook_{letter}")
      circuit.add_edge(f"blocks.{layer}.hook_{letter}_input", f"blocks.{layer}.attn.hook_{letter}")

    circuit.add_node(f"blocks.{layer}.attn.hook_result")
    for letter in "qkv":
      circuit.add_edge(f"blocks.{layer}.attn.hook_{letter}", f"blocks.{layer}.attn.hook_result")

    current_nodes = list(circuit.nodes)
    nodes_that_receive_resid_directly = [f"blocks.{layer}.hook_q_input",
                                         f"blocks.{layer}.hook_k_input",
                                         f"blocks.{layer}.hook_v_input"]
    for from_node in current_nodes:
      if any([from_node.endswith(resid_writer) for resid_writer in resid_writers]):
        if f"blocks.{layer}" not in from_node:
          for to_node in nodes_that_receive_resid_directly:
            circuit.add_edge(from_node, to_node)

    # mlp
    circuit.add_node(f"blocks.{layer}.hook_mlp_in")
    circuit.add_node(f"blocks.{layer}.hook_mlp_out")
    circuit.add_edge(f"blocks.{layer}.hook_mlp_in", f"blocks.{layer}.hook_mlp_out")

    current_nodes = list(circuit.nodes)
    nodes_that_receive_resid_directly = [f"blocks.{layer}.hook_mlp_in"]
    for from_node in current_nodes:
      if any([from_node.endswith(resid_writer) for resid_writer in resid_writers]):
        if (f"blocks.{layer}" not in from_node) or ("attn.hook_result" in from_node):
          for to_node in nodes_that_receive_resid_directly:
            circuit.add_edge(from_node, to_node)

  circuit.add_node(f"blocks.{n_layers - 1}.hook_resid_post")
  for from_node in list(circuit.nodes):
    if any([from_node.endswith(resid_writer) for resid_writer in resid_writers]):
      circuit.add_edge(from_node, f"blocks.{n_layers - 1}.hook_resid_post")

  return circuit