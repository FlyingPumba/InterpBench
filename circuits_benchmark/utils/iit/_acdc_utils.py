from typing import Tuple

from circuits_benchmark.benchmark.benchmark_case import BenchmarkCase
from circuits_benchmark.utils.circuit.circuit import Circuit, CircuitNode
from iit.model_pairs.nodes import LLNode
from iit.utils import index
from iit.utils.correspondence import Correspondence
from circuits_benchmark.utils.circuit.prepare_circuit import prepare_circuit_for_evaluation


def convert_LLNode_to_CircuitNodes(
    ll_node: LLNode, n_heads: int
) -> list[CircuitNode]:
    """
    Splits the LLNode index to create multiple CircuitNode objects for each of them.
    No sanity checks are applied on the node name.
    """
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

    for i in get_circuit_idxs_from_ll_idx(ll_node.index):
        circuit_nodes.append(CircuitNode(ll_node.name, i))
    return circuit_nodes

def find_corresponding_ll_node(
    hl_node: CircuitNode, hl_ll_corr: Correspondence
) -> set[LLNode] | None:
    for k, v in hl_ll_corr.items():
        if k == hl_node:
            return v
    return None


def map_tracr_edges_to_ll_edges(
    tracr_circuit: Circuit,
    hl_ll_corr: Correspondence,
    n_heads: int,
    tracr_leaf_node: CircuitNode,
    ll_leaf_node: CircuitNode,
) -> list[Tuple[CircuitNode, CircuitNode]]:
    """
    A whacky method to find the corresponding edges in the 'low-level' circuit,
    given tracr edges and a correspondence between them. 
    It also accounts for mismatch in the number of layers between them by mapping all edges
    that point to tracr's leaf node to the low-level leaf node. 
    Args:
        tracr_circuit: The tracr circuit
        edge: The edge to find
        hl_ll_corr: The correspondence between high-level and low-level nodes
        n_heads: Number of heads
        tracr_leaf_node: The leaf node of the tracr circuit
        ll_leaf_node: The leaf node of the low-level circuit
    """
    all_edges = []
    for e in tracr_circuit.edges:
        # By default, the edge is the same as in tracr.
        # This is useful when the hl_ll_corr maps only a subset of the tracr circuit.
        # (I don't know why we are allowing this, but we need to do this for now.)
        new_ll_from_nodes = [e[0]]
        new_ll_to_nodes = [e[1]]

        # make corresponding ll edges using hl_ll_corr
        v = find_corresponding_ll_node(e[0], hl_ll_corr)
        if v is not None:
            new_ll_from_nodes = []
            for node in v:
                new_ll_from_nodes.extend(
                    convert_LLNode_to_CircuitNodes(node, n_heads)
                )

        v = find_corresponding_ll_node(e[1], hl_ll_corr)
        if v is not None:
            new_ll_to_nodes = []
            for node in v:
                new_ll_to_nodes.extend(
                    convert_LLNode_to_CircuitNodes(node, n_heads)
                )
        
        # if the tracr's edge points to the leaf node, map it to the ll_leaf_node
        if v is None and e[1] == tracr_leaf_node:
            new_ll_to_nodes = [ll_leaf_node]
        
        # make all possible edges
        new_edges = [(f, t) for f in new_ll_from_nodes for t in new_ll_to_nodes]
        all_edges.extend(new_edges)
    return all_edges


def get_gt_circuit(
    hl_ll_corr: Correspondence,
    full_circuit: Circuit,
    n_heads: int,
    case: BenchmarkCase,
    promote_to_heads: bool = True,
) -> Circuit:
    """
    Makes a circuit for the ll_model using tracr's ground truth circuit and the correspondence.
    """
    if not promote_to_heads:
        raise NotImplementedError("Only promote_to_heads=True is supported")
    
    circuit = full_circuit.copy()
    circuit = prepare_circuit_for_evaluation(circuit, promote_to_heads)
    circuit_leaf_node = circuit.get_result_node()

    # remove edges that are not a part of the tracr ground truth
    tracr_ll_circuit = case.get_ll_gt_circuit(granularity="acdc_hooks")
    tracr_ll_circuit = prepare_circuit_for_evaluation(tracr_ll_circuit, promote_to_heads)
    tracr_leaf_node = tracr_ll_circuit.get_result_node()

    edges_to_keep = map_tracr_edges_to_ll_edges(
        tracr_ll_circuit, hl_ll_corr, n_heads, tracr_leaf_node, circuit_leaf_node
    )
    edges_to_remove = set(circuit.edges) - set(edges_to_keep)

    for edge in edges_to_remove:
        # print(f"Removing edge {edge}")
        circuit.remove_edge(edge[0], edge[1])

    # remove detached nodes
    nodes_to_remove = set()
    for node in circuit.nodes:
        if not list(circuit.successors(node)) and node != circuit_leaf_node:
            assert list(circuit.predecessors(node)) == [], RuntimeError("Found two leaf nodes in he circuit. This should not happen.")
            nodes_to_remove.add(node)
    
    for node in nodes_to_remove:
        circuit.remove_node(node)
    
    assert prepare_circuit_for_evaluation(circuit, promote_to_heads).nodes == circuit.nodes, RuntimeError("Some nodes were not removed from the circuit")
    assert prepare_circuit_for_evaluation(circuit, promote_to_heads).edges == circuit.edges, RuntimeError("Some edges were not removed from the circuit")
    assert len(circuit.edges) > 0, RuntimeError("No edges left in the circuit")
    assert len(circuit.nodes) > 0, RuntimeError("No nodes left in the circuit")
    return circuit
