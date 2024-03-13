import iit.model_pairs as mp
from circuits_benchmark.transformers.circuit import Circuit
from circuits_benchmark.transformers.circuit import Circuit
import iit.model_pairs as mp
from circuits_benchmark.transformers.hooked_tracr_transformer import HookedTracrTransformer
from ._acdc_utils import get_circuit_nodes_from_ll_node, get_gt_circuit
from circuits_benchmark.transformers.acdc_circuit_builder import get_full_acdc_circuit
import iit_utils.correspondence as correspondence

def evaluate_acdc_circuit(
        acdc_circuit: Circuit,
        ll_model: HookedTracrTransformer,
        hl_ll_corr: correspondence.TracrCorrespondence,
        verbose: bool = False
):
    full_circuit = get_full_acdc_circuit(ll_model.cfg.n_layers, ll_model.cfg.n_heads)
    gt_circuit = get_gt_circuit(hl_ll_corr, full_circuit)
    return calculate_fpr_and_tpr(acdc_circuit, gt_circuit, full_circuit, verbose)


def calculate_fpr_and_tpr(
        acdc_circuit: Circuit, 
        gt_circuit: Circuit, 
        full_circuit: Circuit, 
        verbose: bool = False
):
    all_nodes = set(full_circuit.nodes)
    all_edges = set(full_circuit.edges)

    # calculate nodes false positives and false negatives
    acdc_nodes = set(acdc_circuit.nodes)
    tracr_nodes = set(gt_circuit.nodes)
    false_positive_nodes = acdc_nodes - tracr_nodes
    false_negative_nodes = tracr_nodes - acdc_nodes
    true_positive_nodes = acdc_nodes & tracr_nodes
    true_negative_nodes = all_nodes - (acdc_nodes | tracr_nodes)

    if verbose:
        print("\nNodes analysis:")
        print(f" - False Positives: {sorted(false_positive_nodes)}")
        print(f" - False Negatives: {sorted(false_negative_nodes)}")
        print(f" - True Positives: {sorted(true_positive_nodes)}")
        print(f" - True Negatives: {sorted(true_negative_nodes)}")

    # calculate edges false positives and false negatives
    acdc_edges = set(acdc_circuit.edges)
    tracr_edges = set(gt_circuit.edges)
    false_positive_edges = acdc_edges - tracr_edges
    false_negative_edges = tracr_edges - acdc_edges
    true_positive_edges = acdc_edges & tracr_edges
    true_negative_edges = all_edges - (
        acdc_edges | tracr_edges
    )  # == (all_edges - acdc_edges) & (all_edges - tracr_edges)

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
