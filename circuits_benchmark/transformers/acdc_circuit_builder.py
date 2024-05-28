from acdc.TLACDCCorrespondence import TLACDCCorrespondence
from circuits_benchmark.transformers.circuit import Circuit
from circuits_benchmark.transformers.circuit_node import CircuitNode
from acdc.TLACDCEdge import EdgeType


def build_acdc_circuit(corr: TLACDCCorrespondence) -> Circuit:
    circuit = Circuit()

    for (child_name, child_index, parent_name, parent_index), edge in corr.edge_dict().items():
        if edge.present and edge.edge_type != EdgeType.PLACEHOLDER:
            parent_head_index = None
            if (
                parent_index is not None
                and len(parent_index.hashable_tuple) > 2
                and parent_index.hashable_tuple[2] is not None
            ):
                parent_head_index = parent_index.hashable_tuple[2]

            child_head_index = None
            if (
                child_index is not None
                and len(child_index.hashable_tuple) > 2
                and child_index.hashable_tuple[2] is not None
            ):
                child_head_index = child_index.hashable_tuple[2]

            from_node = CircuitNode(parent_name, parent_head_index)
            to_node = CircuitNode(child_name, child_head_index)
            circuit.add_edge(from_node, to_node)

    return circuit


def get_full_acdc_circuit(n_layers: int, n_heads: int) -> Circuit:
    circuit = Circuit()

    circuit.add_node(CircuitNode("hook_embed"))
    circuit.add_node(CircuitNode("hook_pos_embed"))

    # nodes that write to residual stream match at least one of the following
    resid_writers_filter = ["hook_embed", "hook_pos_embed", "attn.hook_result", "hook_mlp_out"]

    for layer in range(n_layers):
        upstream_nodes = list(circuit.nodes)

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

            nodes_that_receive_resid_directly = [
                f"blocks.{layer}.hook_q_input",
                f"blocks.{layer}.hook_k_input",
                f"blocks.{layer}.hook_v_input",
            ]
            for from_node in upstream_nodes:
                if any([from_node.name.endswith(resid_writer) for resid_writer in resid_writers_filter]):
                    if f"blocks.{layer}" not in from_node.name:  # discard nodes from current layer
                        for to_node_name in nodes_that_receive_resid_directly:
                            circuit.add_edge(from_node, CircuitNode(to_node_name, head))

        upstream_nodes = list(circuit.nodes)

        # mlp
        mlp_in_node = CircuitNode(f"blocks.{layer}.hook_mlp_in")
        mlp_out_node = CircuitNode(f"blocks.{layer}.hook_mlp_out")
        circuit.add_node(mlp_in_node)
        circuit.add_node(mlp_out_node)
        circuit.add_edge(mlp_in_node, mlp_out_node)

        nodes_that_receive_resid_directly = [mlp_in_node]
        for from_node in upstream_nodes:
            if any([from_node.name.endswith(resid_writer) for resid_writer in resid_writers_filter]):
                if (f"blocks.{layer}" not in from_node.name) or ("attn.hook_result" in from_node.name):
                    for to_node in nodes_that_receive_resid_directly:
                        circuit.add_edge(from_node, to_node)

    last_resid_post_node = CircuitNode(f"blocks.{n_layers - 1}.hook_resid_post")
    circuit.add_node(last_resid_post_node)
    for from_node in list(circuit.nodes):
        if any([from_node.name.endswith(resid_writer) for resid_writer in resid_writers_filter]):
            circuit.add_edge(from_node, last_resid_post_node)

    return circuit


def replace_inputs_and_qkv_nodes_with_outputs(circuit: Circuit) -> Circuit:
    new_circuit = Circuit()
    prefix = lambda node_name: ".".join(node_name.split(".")[:-1])
    suffix = lambda node_name: node_name.split(".")[-1]

    for node in circuit.nodes:
        node_name_prefix = prefix(node.name)
        node_name_suffix = suffix(node.name)
        if node_name_suffix in ["hook_q_input", "hook_k_input", "hook_v_input"]:
            new_node = CircuitNode(f"{node_name_prefix}.attn.hook_result", node.index)
        elif node_name_suffix in ["hook_embed", "hook_pos_embed", "hook_resid_pre"]:
            continue
        elif node_name_suffix == "hook_mlp_in":
            # TODO:
            # We are doing this because SP removes this type of direct computation edge
            # this needs to be fixed in the SP code
            new_node = CircuitNode(f"{node_name_prefix}.hook_mlp_out")
        elif node_name_suffix in ["hook_q", "hook_k", "hook_v"]:
            new_node = CircuitNode(f"{node_name_prefix}.hook_result", node.index)
        else:
            new_node = node
        new_circuit.add_node(new_node)

    return new_circuit.nodes


def replace_inputs_and_qkv_edges_with_outputs(circuit: Circuit) -> Circuit:
    new_circuit = Circuit()
    prefix = lambda node_name: ".".join(node_name.split(".")[:-1])
    suffix = lambda node_name: node_name.split(".")[-1]

    for from_node, to_node in circuit.edges:
        from_node_name_prefix = prefix(from_node.name)
        from_node_name_suffix = suffix(from_node.name)
        to_node_name_prefix = prefix(to_node.name)
        to_node_name_suffix = suffix(to_node.name)
        qkv_ins = ["hook_q_input", "hook_k_input", "hook_v_input"]
        qkv_outs = ["hook_q", "hook_k", "hook_v"]
        resids = ["hook_resid_post", "hook_resid_pre"] # TODO: maybe not the last one

        if ( 
           (from_node_name_suffix in qkv_ins)  # Placeholder: {qkv}_input -> hook_{qkv}
        or (from_node_name_suffix in qkv_outs)  # Direct computation: hook_q -> hook_result
        or (from_node_name_suffix == "hook_mlp_in") # Direct computation: hook_mlp_in -> hook_mlp_out
        ):
            # Ignore direct computation and placeholder edges
            if (
                (to_node_name_suffix in qkv_outs)
                or (to_node_name_suffix == "hook_result")
                or (to_node_name_suffix == "hook_mlp_out")
            ):
                print(
                    f"!!! WARNING: Received an innvalid edge:", 
                    f"{from_node.name} -> {to_node.name}"
                )
            continue  
        elif from_node_name_suffix in resids or to_node_name_suffix in resids:
            # ignore resid edges: TODO: Not sure if I should consider these edges...
            continue
        elif to_node_name_suffix in qkv_ins:
            # directly route incoming edges to head's hook_result
            new_from_node = from_node
            new_to_node = CircuitNode(f"{to_node_name_prefix}.attn.hook_result", to_node.index)
        elif to_node_name_suffix == "hook_mlp_in":
            # directly route incoming edges to mlp_out
            new_from_node = from_node
            new_to_node = CircuitNode(f"{to_node_name_prefix}.hook_mlp_out")
        else:
            # all other edges are okay...
            new_from_node = from_node
            new_to_node = to_node
        new_circuit.add_edge(new_from_node, new_to_node)
    return new_circuit.edges
