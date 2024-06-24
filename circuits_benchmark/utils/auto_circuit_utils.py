from auto_circuit.types import PruneScores
from auto_circuit.utils.patchable_model import PatchableModel

from circuits_benchmark.utils.circuit.circuit import Circuit
from circuits_benchmark.utils.circuit.circuit_node import CircuitNode


def build_circuit(model: PatchableModel,
                  attribution_scores: PruneScores,
                  threshold: float) -> Circuit:
  """Build a circuit out of the auto_circuit output."""
  circuit = Circuit()

  for edge in model.edges:
    src_node = edge.src
    dst_node = edge.dest
    score = attribution_scores[dst_node.module_name][edge.patch_idx]

    if score >= threshold:
      from_node = CircuitNode(src_node.module_name, src_node.head_idx)
      to_node = CircuitNode(dst_node.module_name, dst_node.head_idx)
      circuit.add_edge(from_node, to_node)

  return circuit
