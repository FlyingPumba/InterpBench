from circuits_benchmark.transformers.circuit import Circuit
from ._acdc_utils import get_gt_circuit
from circuits_benchmark.transformers.acdc_circuit_builder import get_full_acdc_circuit
from circuits_benchmark.transformers.hooked_tracr_transformer import HookedTracrTransformer
from circuits_benchmark.utils.get_cases import get_cases
from circuits_benchmark.commands.build_main_parser import build_main_parser
import circuits_benchmark.utils.iit.correspondence as correspondence
from argparse import Namespace
import os
import shutil
import circuits_benchmark.commands.algorithms.acdc as acdc
import pickle

def evaluate_acdc_circuit(
        acdc_circuit: Circuit,
        ll_model: HookedTracrTransformer,
        hl_ll_corr: correspondence.TracrCorrespondence,
        verbose: bool = False
):
    full_circuit = get_full_acdc_circuit(ll_model.cfg.n_layers, ll_model.cfg.n_heads)
    gt_circuit = get_gt_circuit(hl_ll_corr, full_circuit, ll_model.cfg.n_heads)
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

def run_acdc_eval(case_num, weight, threshold, using_wandb=False):
    args = Namespace(
        command="compile",
        indices=f"{case_num}",
        output_dir="/Users/cybershiptrooper/src/interpretability/MATS/circuits-benchmark/results/compile",
        device="cpu",
        seed=1234,
        run_tests=False,
        tests_atol=1e-05,
        fail_on_error=False,
        original_args=["compile", f"-i={case_num}", "-f"],
    )
    
    cases = get_cases(args)
    case = cases[0]
    if not case.supports_causal_masking():
        raise NotImplementedError(f"Case {case.get_index()} does not support causal masking")

    tracr_output = case.build_tracr_model()
    hl_model = case.build_transformer_lens_model()

    metric = "l2" if not hl_model.is_categorical() else "kl"

    # this is the graph node -> hl node correspondence
    # tracr_hl_corr = correspondence.TracrCorrespondence.from_output(tracr_output)
    output_suffix = f"weight_{weight}/threshold_{threshold}"
    clean_dirname = f"results/acdc_{case.get_index()}/{output_suffix}"
    # remove everything in the directory
    if os.path.exists(clean_dirname):
        shutil.rmtree(clean_dirname)
    cfg_dict = {
        "n_layers": 2,
        "n_heads": 4,
        "d_head": 4,
        "d_model": 8,
        "d_mlp": 16,
        "seed": 0,
        "act_fn": "gelu",
    }
    ll_cfg = hl_model.cfg.to_dict().copy()
    ll_cfg.update(cfg_dict)

    ll_model = HookedTracrTransformer(
        ll_cfg, hl_model.tracr_input_encoder, hl_model.tracr_output_encoder, hl_model.residual_stream_labels,
        remove_extra_tensor_cloning=True
    )
    if weight != "tracr":
        ll_model.load_weights_from_file(f"ll_models/{case_num}/ll_model_{weight}.pth")

    print(ll_model.device)
    ll_model.to(ll_model.device)
    for param in ll_model.parameters():
        print(param.device)
        break
    wandb_str = f"--using-wandb" if using_wandb else ""
    acdc_args, _ = build_main_parser().parse_known_args(
        [
            "run",
            "acdc",
            f"--threshold={threshold}",
            f"--metric={metric}",
            wandb_str,
            "--wandb-entity-name=cybershiptrooper",
            f"--wandb-project-name=acdc_{case.get_index()}"

        ]
    )  #'--data_size=1000'])
    if weight == "tracr":
        acdc_circuit, result = acdc.run_acdc(case, acdc_args, calculate_fpr_tpr=True, output_suffix=output_suffix)
    else:
        acdc_circuit, acdc_result = acdc.run_acdc(case, acdc_args, ll_model, calculate_fpr_tpr=False, output_suffix=output_suffix)
        print("Done running acdc: ")
        print(list(acdc_circuit.nodes), list(acdc_circuit.edges))

        # get the ll -> hl correspondence
        hl_ll_corr = correspondence.TracrCorrespondence.from_output(case=case, tracr_output=tracr_output)
        print("hl_ll_corr:", hl_ll_corr)
        hl_ll_corr.save(f"{clean_dirname}/hl_ll_corr.pkl")
        # evaluate the acdc circuit
        print("Calculating FPR and TPR for threshold", threshold)
        result = evaluate_acdc_circuit(acdc_circuit, ll_model, hl_ll_corr, verbose=False)
        result.update(acdc_result)

    # save the result
    with open(f"{clean_dirname}/result.txt", "w") as f:
        f.write(str(result))
    pickle.dump(result, open(f"{clean_dirname}/result.pkl", "wb"))
    print(f"Saved result to {clean_dirname}/result.txt and {clean_dirname}/result.pkl")
    return result

    # metric_name = "l2"
    # validation_metric = case.get_validation_metric(metric_name, ll_model, data_size=1200)
    # from iit_utils.dataset import TracrIITDataset, create_dataset, get_encoded_input_from_torch_input
    # data, _ = create_dataset(case, hl_model, 1200, 0)
    # inputs, outputs, _ = get_encoded_input_from_torch_input(zip(*data.base_data[:]), hl_model, ll_model.device)
    # # print(f"Validation metric: {validation_metric}", "\n\noutputs:", outputs)
    # print(f"Validation metric: {validation_metric(outputs.unsqueeze(-1))}")

    # raise
