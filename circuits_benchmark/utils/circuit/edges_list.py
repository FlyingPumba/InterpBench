from typing import List, Tuple

from circuits_benchmark.utils.circuit.circuit import Circuit
from circuits_benchmark.utils.circuit.circuit_node import CircuitNode


def edges_list_to_circuit(edges: List[Tuple[str, str]]) -> Circuit:
    circuit = Circuit()

    for edge in edges:
        start_node_name = edge[0].split("[")[0]
        end_node_name = edge[1].split("[")[0]

        if "[" in edge[0]:
            start_node_index = int(edge[0].split("[")[1].split("]")[0])
        else:
            start_node_index = None

        if "[" in edge[1]:
            end_node_index = int(edge[1].split("[")[1].split("]")[0])
        else:
            end_node_index = None

        start_node = CircuitNode(start_node_name, start_node_index)
        end_node = CircuitNode(end_node_name, end_node_index)

        circuit.add_edge(start_node, end_node)

    return circuit


def circuit_to_edges_list(circuit: Circuit) -> List[Tuple[str, str]]:
    edges = []

    for edge in circuit.edges:
        start_node_name = edge[0].name
        end_node_name = edge[1].name

        if edge[0].index is not None:
            start_node_name += f"[{edge[0].index}]"

        if edge[1].index is not None:
            end_node_name += f"[{edge[1].index}]"

        edges.append((start_node_name, end_node_name))

    return edges