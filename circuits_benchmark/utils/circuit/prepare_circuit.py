from circuits_benchmark.utils.circuit.circuit import Circuit
from circuits_benchmark.utils.circuit.circuit_node import CircuitNode


def prepare_circuit_for_evaluation(
    circuit: Circuit,
    remove_edges_from_qkv_inputs: bool = True,
    reroute_edges_to_qkv_inputs: bool = True,
    remove_edges_from_qkv_outputs: bool = True,
    remove_edges_from_mlp_in: bool = True,
    reroute_edges_to_mlp_in: bool = True,
    rename_edges_from_embed_to_resid_pre: bool = True,
    remove_embed_to_resid_edges: bool = True,
    remove_ignorable_resid_edges: bool = True,
) -> Circuit:
    """
    Prepare the circuit by removing or rerouting specific edges based on the given arguments.

    Args:
        circuit: The circuit to prepare
        remove_edges_from_qkv_inputs: Remove edges originating from qkv input nodes.
            E.g., removes edges like {qkv}_input -> hook_{qkv}
        reroute_edges_to_qkv_inputs: Reroute edges whose destination is a qkv input node to the corresponding head's hook_result.
            E.g., reroutes edges like hook_mlp_out -> hook_{qkv}_input to hook_mlp_out -> hook_attn_result
        remove_edges_from_mlp_in: Remove edges originating from mlp_in nodes
            E.g., removes edges like hook_mlp_in -> hook_mlp_out
        reroute_edges_to_mlp_in: Reroute edges whose destination is an mlp_in node to the corresponding hook_mlp_out
            E.g., reroutes edges like hook_{pos_embed} -> hook_mlp_in to hook_{pos_embed} -> hook_mlp_out
        rename_edges_from_embed_to_resid_pre: Use hook_resid_pre instead of embed nodes
            E.g., changes edges like hook_embed -> hook_mlp_in to blocks.0.hook_resid_pre -> hook_mlp_in
        remove_embed_to_resid_edges: Remove edges originating from embed nodes and going to resid nodes
            E.g., removes edges like hook_embed -> hook_resid_post
        remove_ignorable_resid_edges: Remove edges originating from resid nodes that are not the first layer and going to resid nodes or from resid nodes to the last layer
            E.g., removes edges like hook_resid_pre -> hook_resid_post

    Returns:
        The prepared circuit
    """
    # Get nodes that are sinks (leafs) in the circuit, such as blocks.{n_layer-1}.hook_resid_post
    sink_nodes = [node for node in circuit.nodes if not list(circuit.successors(node))]

    # Get nodes that are sources (roots) in the circuit, such as hook_embed or hook_pos_embed
    source_nodes = [node for node in circuit.nodes if not list(circuit.predecessors(node))]

    new_circuit = Circuit()
    for from_node, to_node in circuit.edges:
        # Skip the edge based on the removal criteria
        if remove_edges_from_qkv_inputs and is_qkv_input(from_node):
            continue

        if remove_edges_from_qkv_outputs and is_qkv_out(from_node):
            continue

        if remove_edges_from_mlp_in and is_mlp_in(from_node):
            continue

        if remove_embed_to_resid_edges and is_embed(from_node) and is_resid(to_node):
            continue

        if remove_ignorable_resid_edges and is_ignorable_resid_edge(from_node, to_node, sink_nodes, source_nodes):
            continue

        if rename_edges_from_embed_to_resid_pre and is_embed(from_node):
            from_node = CircuitNode("blocks.0.hook_resid_pre")

        if reroute_edges_to_qkv_inputs and is_qkv_input(to_node):
            # directly route incoming edges to head's hook_result
            to_node = CircuitNode(f"{prefix(to_node.name)}.attn.hook_result", to_node.index)
        elif reroute_edges_to_mlp_in and is_mlp_in(to_node):
            # directly route incoming edges to mlp_out
            to_node = CircuitNode(f"{prefix(to_node.name)}.hook_mlp_out")

        new_circuit.add_edge(from_node, to_node)

    return new_circuit


def is_qkv_out(node: CircuitNode) -> bool:
    return suffix(node.name) in ["hook_q", "hook_k", "hook_v"]


def is_qkv_input(node: CircuitNode) -> bool:
    return suffix(node.name) in ["hook_q_input", "hook_k_input", "hook_v_input"]


def is_mlp_in(node: CircuitNode) -> bool:
    return suffix(node.name) == "hook_mlp_in"


def is_embed(node: CircuitNode) -> bool:
    return suffix(node.name) in ["hook_embed", "hook_pos_embed"]


def is_resid(node: CircuitNode) -> bool:
    return suffix(node.name) in ["hook_resid_post", "hook_resid_pre"]


def is_ignorable_resid_edge(from_node: CircuitNode,
                            to_node: CircuitNode,
                            sink_nodes: list[CircuitNode],
                            source_nodes: list[CircuitNode]) -> bool:
    # ignore every edge that comes from resid stream
    # other than edges from first layer (that do not go back to resid stream)
    if is_resid(from_node):
        if from_node in source_nodes and not is_resid(to_node):
            return False
        return True

    # return False when edges are to the last layer
    return to_node not in sink_nodes and is_resid(to_node)


def suffix(node_name: str) -> str:
    return node_name.split(".")[-1]


def prefix(node_name: str) -> str:
    return ".".join(node_name.split(".")[:-1])
