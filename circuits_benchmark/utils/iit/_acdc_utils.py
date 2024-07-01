from typing import Tuple

from circuits_benchmark.benchmark.benchmark_case import BenchmarkCase
from circuits_benchmark.utils.circuit.circuit import Circuit, CircuitNode
from iit.model_pairs.nodes import LLNode
from iit.utils import index
from iit.utils.correspondence import Correspondence


def get_circuit_nodes_from_ll_node(
    ll_node: LLNode, n_heads: int
) -> list[CircuitNode]:
    circuit_nodes = []

    def get_circuit_idxs_from_ll_idx(ll_idx: index.Index) -> list[int]:
        if ll_idx is index.Ix[[None]] or ll_idx is None:
            return [None]
        id = ll_idx.as_index[2]
        if isinstance(id, slice):
            if id.start is None:
                return list(range(id.stop))
            elif id.stop is None:
                return list(range(id.start, n_heads + 1))
            return list(range(id.start, id.stop))
        assert isinstance(id, int), ValueError(
            f"Unexpected index type {type(id)}"
        )
        return [id]

    node_name = ll_node.name
    ll_node_index = ll_node.index
    node_name_type = node_name.split(".")[2]
    node_layer = node_name.split(".")[1]
    if node_name_type == "attn":
        assert node_name.split(".")[-1] == "hook_result"
        # add nodes for q/k/v input, hook_{q/k/v/result}
        prefix = f"blocks.{node_layer}"
        suffixes = [f"hook_{qkv}_input" for qkv in ["q", "k", "v"]]
        suffixes.extend([f"attn.hook_{qkv}" for qkv in ["q", "k", "v"]])
        suffixes.append("attn.hook_result")

        # add all nodes for indices in ll_node_index
        circuit_indices = get_circuit_idxs_from_ll_idx(ll_node_index)
        for circuit_index in circuit_indices:
            for suffix in suffixes:
                circuit_nodes.append(
                    CircuitNode(f"{prefix}.{suffix}", circuit_index)
                )
    else:
        assert ll_node.index == index.Ix[[None]], ValueError(
            f"Node of type {node_name_type} has non-None index {ll_node.index}"
        )
        if node_name_type == "mlp":
            prefix = f"blocks.{node_layer}"
            suffixes = ["hook_mlp_out", "hook_mlp_in"]
            circuit_nodes = [
                CircuitNode(f"{prefix}.{suffix}", None) for suffix in suffixes
            ]
        else:
            circuit_nodes = [CircuitNode(node_name, None)]
    return circuit_nodes


#############################################################
# This only works when we promote things!


def promote_node(node: CircuitNode):
    node_prefix = ".".join(node.name.split(".")[:-1])
    node_suffix = node.name.split(".")[-1]
    if node_suffix in ["hook_k_input", "hook_q_input", "hook_v_input"]:
        node = CircuitNode(f"{node_prefix}.attn.hook_result", node.index)
    elif node_suffix in ["hook_k", "hook_q", "hook_v"]:
        node = CircuitNode(f"{node_prefix}.hook_result", node.index)
    elif node_suffix == "mlp_in":
        node = CircuitNode(f"{node_prefix}.mlp.hook_post", node.index)

    return node


def promote_and_check_with_hl(
    node: CircuitNode, hl_ll_corr: Correspondence
) -> list[CircuitNode]:
    promoted_node = promote_node(node)
    for k, v in hl_ll_corr.items():
        if k == promoted_node:
            return v
    return None


def find_edge_in_circuit(
    tracr_circuit: Circuit,
    edge: Tuple[str, str],
    hl_ll_corr: Correspondence,
    n_heads: int,
) -> bool:
    """
    A whacky method to find the corresponding edge in the 'low-level' circuit,
    given tracr edges and a correspondence between them.
    """
    for e in tracr_circuit.edges:
        # By default, the edge is the same as in tracr.
        # This is useful when the hl_ll_corr maps only a subset of the tracr circuit.
        # (I don't know why we are allowing this, but we need to do this for now.)
        new_ll_from_nodes = [e[0]]
        new_ll_to_nodes = [e[1]]

        # make corresponding ll edges using hl_ll_corr
        v = promote_and_check_with_hl(e[0], hl_ll_corr)
        # I need to promote and check because IIT is only defined for nodewise maps
        # So the corr only has hook_result/hook_mlp_out

        if v is not None:
            new_ll_from_nodes = []
            for node in v:
                new_ll_from_nodes.extend(
                    get_circuit_nodes_from_ll_node(node, n_heads)
                )

        v = promote_and_check_with_hl(e[1], hl_ll_corr)
        if v is not None:
            new_ll_to_nodes = []
            for node in v:
                new_ll_to_nodes.extend(
                    get_circuit_nodes_from_ll_node(node, n_heads)
                )

        new_edges = [(f, t) for f in new_ll_from_nodes for t in new_ll_to_nodes]
        edge = (
            promote_node(edge[0]),
            promote_node(edge[1]),
        )  # promote edge to match the nodes in the new_edges
        if edge in new_edges:
            return True
    return False


def get_gt_circuit(
    hl_ll_corr: Correspondence,
    full_circuit: Circuit,
    n_heads: int,
    case: BenchmarkCase,
) -> Circuit:
    # print(
    #     "WARNING: This method only works when we promote nodes!",
    #     "This has not been tested for qkv granularity. Use with care.",
    # )
    circuit = full_circuit.copy()
    corr_vals = hl_ll_corr.values()
    all_nodes = set(full_circuit.nodes)

    # get circuit nodes from ll nodes
    all_circuit_nodes = set()
    for nodes in corr_vals:
        for ll_node in nodes:
            circuit_nodes = get_circuit_nodes_from_ll_node(ll_node, n_heads)
            all_circuit_nodes.update(circuit_nodes)

    # get additional nodes needed
    additional_nodes_needed = set()
    nodes_to_check = ["resid", "embed", "unembed"]
    for node in all_nodes:
        # check if node.name contains the substrings in nodes_to_check
        if any([substr in node.name for substr in nodes_to_check]):
            additional_nodes_needed.add(node)
    nodes_needed = all_circuit_nodes.union(additional_nodes_needed)

    # remove nodes not needed
    nodes_to_remove = all_nodes - nodes_needed
    for node in nodes_to_remove:
        circuit.remove_node(node)

    # remove edges that are not a part of the tracr ground truth
    tracr_ll_circuit = case.get_ll_gt_circuit(granularity="acdc_hooks")
    edges_to_remove = set()

    for edge in circuit.edges:
        if not find_edge_in_circuit(
            tracr_ll_circuit, edge, hl_ll_corr, n_heads
        ):
            edges_to_remove.add(edge)

    for edge in edges_to_remove:
        # print(f"Removing edge {edge}")
        circuit.remove_edge(edge[0], edge[1])

    return circuit
#############################################################

def build_acdc_circuit_from_list_corr(corr: list) -> Circuit:
    circuit = Circuit()
    raise NotImplementedError("This does not work yet.")

    for corr_item in corr:
        if len(corr_item) == 2:
            corr_item = corr_item[0]
        child_name, child_index, parent_name, parent_index = corr_item
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
