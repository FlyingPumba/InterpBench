from circuits_benchmark.utils.circuit.circuit import Circuit
from circuits_benchmark.utils.circuit.circuit_node import CircuitNode


def prepare_circuit_for_evaluation(circuit: Circuit, promote_to_heads: bool = True) -> Circuit:
    """
    Prepare the circuit by:
    - Promote heads
	- Remove inputs to nodes
	- Replace direct computation and placeholders with a single edge
	- Map embeddings to blocks.0.hook_resid_pre
    - Remove all resid edges other than to resid post or 0 resid pre (may not need this)

    Args:
        circuit: The circuit to prepare
        promote_to_heads: Whether to promote heads

    Returns:
        The prepared circuit
    """
    if not promote_to_heads:
       raise NotImplementedError("This function is not yet tested for promote_to_heads=False")

    new_circuit = Circuit()

    for from_node, to_node in circuit.edges:
        to_node_name_prefix = prefix(to_node.name)
        to_node_name_suffix = suffix(to_node.name)
        qkv_ins = ["hook_q_input", "hook_k_input", "hook_v_input"]
        embeds = ["hook_embed", "hook_pos_embed"]

        print(f"from_node: {from_node.name}, to_node: {to_node.name} \nDCP: {is_direct_computation_or_placeholder_edge(from_node, promote_to_heads)} \nIRE: {is_ignorable_resid_edge(from_node, to_node, circuit)}")
        if (
           (is_direct_computation_or_placeholder_edge(from_node, promote_to_heads))
           or is_ignorable_resid_edge(from_node, to_node, circuit)
        ):
            # Ignore:
            # direct computation and placeholder edges
            # and resid edges that are not from the first and last layer
            continue

        if from_node.name in embeds:
            new_from_node = CircuitNode("blocks.0.hook_resid_pre")
        else:
            new_from_node = from_node

        if to_node_name_suffix in qkv_ins:
            # directly route incoming edges to head's hook_result
            new_to_node = reroute_qkv_in_to_dest(to_node, promote_to_heads)
        elif to_node_name_suffix == "hook_mlp_in":
            # directly route incoming edges to mlp_out
            new_to_node = CircuitNode(f"{to_node_name_prefix}.hook_mlp_out")
        else:
            # all other edges are okay...
            new_to_node = to_node
        new_circuit.add_edge(new_from_node, new_to_node)
    return new_circuit


def reroute_qkv_in_to_dest(node: CircuitNode, promote_to_heads: bool) -> CircuitNode:
    to_node_name_prefix = prefix(node.name)
    if promote_to_heads:
        # reroute incoming edges to head's hook_result
        return CircuitNode(f"{to_node_name_prefix}.attn.hook_result", node.index)
    else:
        # reroute incoming edges to head's hook_{qkv} instead of hook_result
        name_suffix = suffix(node.name)
        is_q_or_k_or_v = "q" if "hook_q" in name_suffix else "k" if "hook_k" in name_suffix else "v"
        return CircuitNode(f"{to_node_name_prefix}.hook_{is_q_or_k_or_v}", node.index)


def is_direct_computation_or_placeholder_edge(from_node: CircuitNode, promote_to_heads: bool) -> bool:
    from_node_name_suffix = suffix(from_node.name)
    qkv_ins = ["hook_q_input", "hook_k_input", "hook_v_input"]
    qkv_outs = ["hook_q", "hook_k", "hook_v"]

    if not promote_to_heads and (from_node_name_suffix in qkv_outs):
        # Do not remove these edges as we need edges from qkv to attn_result
        return False
    
    return (
        (from_node_name_suffix in qkv_ins)  # Placeholder: {qkv}_input -> hook_{qkv}
        or (from_node_name_suffix in qkv_outs)  # Direct computation: hook_q -> hook_result
        or (from_node_name_suffix == "hook_mlp_in")  # Direct computation: hook_mlp_in -> hook_mlp_out
    )

def is_ignorable_resid_edge(from_node: CircuitNode, to_node: CircuitNode, circuit: Circuit) -> bool:
    from_node_name_suffix = suffix(from_node.name)
    to_node_name_suffix = suffix(to_node.name)
    leaf_nodes = [node for node in circuit.nodes if not list(circuit.successors(node))]
    parent_nodes = [node for node in circuit.nodes if not list(circuit.predecessors(node))]

    resids = ["hook_resid_post", "hook_resid_pre"]
    embeds = ["hook_embed", "hook_pos_embed"]

    from_node_in_resid = from_node_name_suffix in resids
    from_node_in_embed = from_node_name_suffix in embeds
    to_node_in_resid = to_node_name_suffix in resids

    # ignore every edge that comes from resid stream 
    # other than edges from first layer (that do not go back to resid stream)
    if from_node_in_resid:
        if from_node in parent_nodes and not to_node_in_resid:
            return False
        return True
    
    # ignore embed to resid edges as they are just input -> output
    if from_node_in_embed and to_node_in_resid:
        return True
    
    # return False when edges are to the last layer
    return (to_node not in leaf_nodes and to_node_in_resid)

def suffix(node_name: str) -> str:
    return node_name.split(".")[-1]

def prefix(node_name: str) -> str:
    return ".".join(node_name.split(".")[:-1])
