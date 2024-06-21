from typing import Optional

from acdc.TLACDCCorrespondence import TLACDCCorrespondence
from circuits_benchmark.benchmark.benchmark_case import BenchmarkCase
from circuits_benchmark.transformers.acdc_circuit_builder import build_acdc_circuit, \
  replace_inputs_and_qkv_nodes_with_outputs, replace_inputs_and_qkv_edges_with_outputs
from circuits_benchmark.transformers.circuit import Circuit
from circuits_benchmark.transformers.hooked_tracr_transformer import HookedTracrTransformer
from circuits_benchmark.utils.iit._acdc_utils import get_gt_circuit
from circuits_benchmark.utils.iit.correspondence import TracrCorrespondence


def calculate_fpr_and_tpr(
    hypothesis_circuit: Circuit,  # e.g., the one provided by ACDC
    true_circuit: Circuit,  # e.g., the one provided by Tracr (ground truth)
    full_circuit: Circuit,
    verbose: bool = False,
    promote_to_heads: bool = True,
    print_summary: bool = True,
):
  if promote_to_heads:
    all_nodes = replace_inputs_and_qkv_nodes_with_outputs(full_circuit)
    true_nodes = replace_inputs_and_qkv_nodes_with_outputs(true_circuit)
    hypothesis_nodes = replace_inputs_and_qkv_nodes_with_outputs(hypothesis_circuit)
  else:
    all_nodes = full_circuit.nodes
    true_nodes = true_circuit.nodes
    hypothesis_nodes = hypothesis_circuit.nodes

  all_nodes = set(all_nodes)

  # calculate nodes false positives and false negatives
  hypothesis_nodes = set(hypothesis_nodes)
  true_nodes = set(true_nodes)

  assert hypothesis_nodes.issubset(
    all_nodes), f"hypothesis nodes contain the following nodes that are not in the full circuit: {hypothesis_nodes - all_nodes}"
  assert true_nodes.issubset(
    all_nodes), f"true nodes contain the following nodes that are not in the full circuit: {true_nodes - all_nodes}"

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
  if promote_to_heads:
    hypothesis_edges = replace_inputs_and_qkv_edges_with_outputs(hypothesis_circuit)
    true_edges = replace_inputs_and_qkv_edges_with_outputs(true_circuit)
    all_edges = replace_inputs_and_qkv_edges_with_outputs(full_circuit)
  else:
    hypothesis_edges = hypothesis_circuit.edges
    true_edges = true_circuit.edges
    all_edges = full_circuit.edges

  hypothesis_edges = set(hypothesis_edges)
  true_edges = set(true_edges)
  all_edges = set(all_edges)

  assert hypothesis_edges.issubset(
    all_edges), f"hypothesis edges contain the following edges that are not in the full circuit: {hypothesis_edges - all_edges}, hypothesis edges: {hypothesis_edges}, all edges: {all_edges}"
  assert true_edges.issubset(
    all_edges), f"true edges contain the following edges that are not in the full circuit: {true_edges - all_edges}"

  false_positive_edges = (hypothesis_edges - true_edges) & all_edges
  false_negative_edges = true_edges - hypothesis_edges
  true_positive_edges = hypothesis_edges & true_edges
  true_negative_edges = all_edges - (
        hypothesis_edges | true_edges)  # == (all_edges - hypothesis_edges) & (all_edges - true_edges)

  if verbose:
    print("\nEdges analysis:")
    print(f" - False Positives: {sorted(false_positive_edges)}")
    print(f" - False Negatives: {sorted(false_negative_edges)}")
    print(f" - True Positives: {sorted(true_positive_edges)}")
    print(f" - True Negatives: {sorted(true_negative_edges)}")

  # print FP and TP rates for nodes and edges as summary
  make_summary = lambda *args, **kwargs: print(*args, **kwargs) if print_summary else None
  if verbose:
    make_summary("\n\n-------------------\n\nhypothesis_edges", hypothesis_edges, "\n-----------\n")
    make_summary("true_edges", true_edges, "\n-----------\n")
    make_summary("all_edges", all_edges, "\n\n-------------------\n\n")
  make_summary(f"\nSummary:")

  if len(true_positive_nodes | false_negative_nodes) == 0:
    nodes_tpr = "N/A"
    make_summary(f" - Nodes TP rate: N/A")
  else:
    nodes_tpr = len(true_positive_nodes) / len(true_positive_nodes | false_negative_nodes)
    make_summary(f" - Nodes TP rate: {nodes_tpr}")

  if len(false_positive_nodes | true_negative_nodes) == 0:
    nodes_fpr = "N/A"
    make_summary(f" - Nodes FP rate: N/A")
  else:
    nodes_fpr = len(false_positive_nodes) / len(false_positive_nodes | true_negative_nodes)
    make_summary(f" - Nodes FP rate: {nodes_fpr}")

  if len(true_positive_edges | false_negative_edges) == 0:
    edges_tpr = "N/A"
    make_summary(f" - Edges TP rate: N/A")
  else:
    edges_tpr = len(true_positive_edges) / len(true_positive_edges | false_negative_edges)
    make_summary(f" - Edges TP rate: {edges_tpr}")

  if len(false_positive_edges | true_negative_edges) == 0:
    edges_fpr = "N/A"
    make_summary(f" - Edges FP rate: N/A")
  else:
    edges_fpr = len(false_positive_edges) / len(false_positive_edges | true_negative_edges)
    make_summary(f" - Edges FP rate: {edges_fpr}")

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

def evaluate_hypothesis_circuit(
    hypothesis_circuit: Circuit,
    ll_model: HookedTracrTransformer,
    hl_ll_corr: TracrCorrespondence,
    case: BenchmarkCase,
    full_circuit: Optional[Circuit] = None,
    use_embeddings: bool = True,
    **kwargs,
):
  if full_circuit is None:
    full_corr = TLACDCCorrespondence.setup_from_model(
      ll_model, use_pos_embed=use_embeddings
    )
    full_circuit = build_acdc_circuit(corr=full_corr)

  gt_circuit = get_gt_circuit(hl_ll_corr, full_circuit, ll_model.cfg.n_heads, case)

  return calculate_fpr_and_tpr(
    hypothesis_circuit, gt_circuit, full_circuit, **kwargs
  )
