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

    if score > threshold:
      from_node = CircuitNode(src_node.module_name, src_node.head_idx)
      to_node = CircuitNode(dst_node.module_name, dst_node.head_idx)
      circuit.add_edge(from_node, to_node)

  return circuit

def build_normalized_scores(attribution_scores: PruneScores) -> PruneScores:
  """Normalize the scores so that they all lie between 0 and 1."""
  max_score = max(scores.max() for scores in attribution_scores.values())
  min_score = min(scores.min() for scores in attribution_scores.values())

  normalized_scores = attribution_scores.copy()
  for module_name, scores in normalized_scores.items():
    normalized_scores[module_name] = (normalized_scores[module_name] - min_score) / (max_score - min_score)

  return normalized_scores
