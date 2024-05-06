from circuits_benchmark.transformers.circuit import Circuit


def calculate_fpr_and_tpr(
    hypothesis_circuit: Circuit, # e.g., the one provided by ACDC
    true_circuit: Circuit,  # e.g., the one provided by Tracr (ground truth)
    full_circuit: Circuit,
    verbose: bool = False
):
  all_nodes = set(full_circuit.nodes)
  all_edges = set(full_circuit.edges)

  # calculate nodes false positives and false negatives
  hypothesis_nodes = set(hypothesis_circuit.nodes)
  true_nodes = set(true_circuit.nodes)
  false_positive_nodes = hypothesis_nodes - true_nodes
  false_negative_nodes = true_nodes - hypothesis_nodes
  true_positive_nodes = hypothesis_nodes & true_nodes
  true_negative_nodes = all_nodes - (hypothesis_nodes | true_nodes)

  if verbose:
    print("\nNodes analysis:")
    print(f" - False Positives: {sorted(false_positive_nodes)}")
    print(f" - False Negatives: {sorted(false_negative_nodes)}")
    print(f" - True Positives: {sorted(true_positive_nodes)}")
    print(f" - True Negatives: {sorted(true_negative_nodes)}")

  # calculate edges false positives and false negatives
  hypothesis_edges = set(hypothesis_circuit.edges)
  true_edges = set(true_circuit.edges)
  false_positive_edges = hypothesis_edges - true_edges
  false_negative_edges = true_edges - hypothesis_edges
  true_positive_edges = hypothesis_edges & true_edges
  true_negative_edges = all_edges - (hypothesis_edges | true_edges) # == (all_edges - hypothesis_edges) & (all_edges - true_edges)

  if verbose:
    print("\nEdges analysis:")
    print(f" - False Positives: {sorted(false_positive_edges)}")
    print(f" - False Negatives: {sorted(false_negative_edges)}")
    print(f" - True Positives: {sorted(true_positive_edges)}")
    print(f" - True Negatives: {sorted(true_negative_edges)}")

  # print FP and TP rates for nodes and edges as summary
  print(f"\nSummary:")

  if len(true_positive_nodes | false_negative_nodes) == 0:
    nodes_tpr = "N/A"
    print(f" - Nodes TP rate: N/A")
  else:
    nodes_tpr = len(true_positive_nodes) / len(true_positive_nodes | false_negative_nodes)
    print(f" - Nodes TP rate: {nodes_tpr}")

  if len(false_positive_nodes | true_negative_nodes) == 0:
    nodes_fpr = "N/A"
    print(f" - Nodes FP rate: N/A")
  else:
    nodes_fpr = len(false_positive_nodes) / len(false_positive_nodes | true_negative_nodes)
    print(f" - Nodes FP rate: {nodes_fpr}")

  if len(true_positive_edges | false_negative_edges) == 0:
    edges_tpr = "N/A"
    print(f" - Edges TP rate: N/A")
  else:
    edges_tpr = len(true_positive_edges) / len(true_positive_edges | false_negative_edges)
    print(f" - Edges TP rate: {edges_tpr}")

  if len(false_positive_edges | true_negative_edges) == 0:
    edges_fpr = "N/A"
    print(f" - Edges FP rate: N/A")
  else:
    edges_fpr = len(false_positive_edges) / len(false_positive_edges | true_negative_edges)
    print(f" - Edges FP rate: {edges_fpr}")

  return {
    "nodes": {
      "true_positive": true_positive_nodes,
      "false_positive": false_positive_nodes,
      "false_negative": false_negative_nodes,
      "true_negative": true_negative_nodes,
      "tpr": nodes_tpr,
      "fpr": nodes_fpr,
    },
    "edges": {
      "true_positive": true_positive_edges,
      "false_positive": false_positive_edges,
      "false_negative": false_negative_edges,
      "true_negative": true_negative_edges,
      "tpr": edges_tpr,
      "fpr": edges_fpr,
    },
  }