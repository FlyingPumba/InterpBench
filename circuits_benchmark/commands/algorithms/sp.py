from circuits_benchmark.commands.common_args import add_common_args
from circuits_benchmark.benchmark.benchmark_case import BenchmarkCase
from circuits_benchmark.transformers.hooked_tracr_transformer import HookedTracrTransformer
from typing import Optional
import sys
import torch
import shutil
from acdc.docstring.utils import AllDataThings
import os
from subnetwork_probing.masked_transformer import EdgeLevelMaskedTransformer
from subnetwork_probing.train import NodeLevelMaskedTransformer

# from acdc.acdc_utils import kl_divergence
from functools import partial
from circuits_benchmark.utils.edge_sp import train_edge_sp, save_edges
from circuits_benchmark.utils.node_sp import train_sp
from circuits_benchmark.metrics.validation_metrics import l2_metric
from circuits_benchmark.transformers.acdc_circuit_builder import build_acdc_circuit, get_full_acdc_circuit
import wandb
from circuits_benchmark.utils.circuits_comparison import calculate_fpr_and_tpr
from circuits_benchmark.utils.iit._acdc_utils import get_gt_circuit
from circuits_benchmark.utils.iit.correspondence import TracrCorrespondence
from circuits_benchmark.transformers.circuit import Circuit
from acdc.TLACDCCorrespondence import TLACDCCorrespondence
from subnetwork_probing.train import iterative_correspondence_from_mask
from circuits_benchmark.commands.evaluation.iit.iit_acdc_eval import evaluate_acdc_circuit


def setup_args_parser(subparsers):
    parser = subparsers.add_parser("sp")
    add_common_args(parser)

    parser.add_argument("--using-wandb", type=bool, default=False)
    parser.add_argument("--wandb-project", type=str, default="subnetwork-probing")
    parser.add_argument("--wandb-entity", type=str, required=False)
    parser.add_argument("--wandb-group", type=str, required=False)
    parser.add_argument("--wandb-dir", type=str, default="/tmp/wandb")
    parser.add_argument("--wandb-mode", type=str, default="online")
    parser.add_argument("--wandb-run-name", type=str, required=False, default=None, help="Value for wandb_run_name")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--loss-type", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--lambda-reg", type=float, default=1)
    parser.add_argument("--zero-ablation", type=int, default=0)
    parser.add_argument("--data-size", type=int, default=10)
    parser.add_argument("--metric", type=str, choices=["l2", "kl"], default="l2")
    parser.add_argument("--edgewise", type=bool, default=False)
    parser.add_argument("--num-examples", type=int, default=50)
    parser.add_argument("--seq-len", type=int, default=300)
    parser.add_argument("--n-loss-average-runs", type=int, default=4)
    parser.add_argument(
        "--torch-num-threads",
        type=int,
        default=0,
        help="How many threads to use for torch (0=all)",
    )
    parser.add_argument("--reset-subject", type=int, default=0)
    # parser.add_argument("--torch-num-threads", type=int, default=0)
    parser.add_argument("--print-stats", type=int, default=1, required=False)
    parser.add_argument("--print-every", type=int, default=1, required=False)


def eval_fn(
        corr: TLACDCCorrespondence,
        ll_model: HookedTracrTransformer,
        hl_ll_corr: TracrCorrespondence,
):
    sp_circuit = build_acdc_circuit(corr=corr)
    return evaluate_acdc_circuit(sp_circuit, ll_model, hl_ll_corr, verbose=False, print_summary=False)

def run_sp(
    case: BenchmarkCase,
    args,
    calculate_fpr_tpr: bool = True,
    output_suffix: str = "",
):
    print(args)
    hl_model = case.build_transformer_lens_model(remove_extra_tensor_cloning=False)
    hl_ll_corr = TracrCorrespondence.from_output(case, tracr_output=case.build_tracr_model())

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

    tl_model = HookedTracrTransformer(
        ll_cfg,
        hl_model.tracr_input_encoder,
        hl_model.tracr_output_encoder,
        hl_model.residual_stream_labels,
        remove_extra_tensor_cloning=False,
    )
    tl_model.to(args.device)
    tl_model.load_weights_from_file(f"{args.output_dir}/ll_models/{case.get_index()}/ll_model_510.pth")

    tags = [f"case{case.get_index()}", "acdc"]
    notes = f"Command: {' '.join(sys.argv)}"

    # Check that dot program is in path
    if not shutil.which("dot"):
        raise ValueError("dot program not in path, cannot generate graphs for ACDC.")

    if args.torch_num_threads > 0:
        torch.set_num_threads(args.torch_num_threads)

    metric_name = args.metric
    zero_ablation = True if args.zero_ablation else False
    using_wandb = True if args.using_wandb else False
    # wandb_entity_name = args.wandb_entity_name
    # wandb_project_name = args.wandb_project_name
    wandb_run_name = args.wandb_run_name
    # wandb_group_name = args.wandb_group_name
    # indices_mode = args.indices_mode
    # names_mode = args.names_mode
    device = args.device
    edgewise = True  # args.edgewise
    # single_step = True if args.single_step else False

    # second_metric = None  # some tasks only have one metric
    # use_pos_embed = True  # Always true for all tracr models.

    data_size = args.data_size
    # data_size = 100
    base = case.get_clean_data(count=int(1.2 * data_size))
    source = case.get_corrupted_data(count=int(1.2 * data_size))
    toks_int_values = base.get_inputs()
    toks_int_labels = base.get_correct_outputs()
    toks_int_values_other = source.get_inputs()
    toks_int_labels_other = source.get_correct_outputs()

    with torch.no_grad():
        baseline_output = tl_model(toks_int_values[:data_size])
        test_baseline_output = tl_model(toks_int_values[data_size:])

    if metric_name == "l2":
        validation_metric = partial(
            l2_metric, baseline_output=baseline_output, is_categorical=tl_model.is_categorical()
        )
        test_metric = partial(l2_metric, baseline_output=test_baseline_output, is_categorical=tl_model.is_categorical())
    else:
        raise NotImplementedError(f"Metric {metric_name} not implemented")

    all_task_things = AllDataThings(
        tl_model=tl_model,
        validation_metric=validation_metric,
        validation_data=toks_int_values[:data_size],
        validation_labels=toks_int_labels[:data_size],
        validation_mask=None,
        validation_patch_data=toks_int_values_other[:data_size],
        test_metrics={"loss": test_metric} if not edgewise else test_metric,
        test_data=toks_int_values[data_size:],
        test_labels=toks_int_labels[data_size:],
        test_mask=None,
        test_patch_data=toks_int_values_other[data_size:],
    )

    output_dir = os.path.join(args.output_dir, f"results/sp_{case.get_index()}", output_suffix)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    images_output_dir = os.path.join(output_dir, f"results/sp_{case.get_index()}", "images")
    if not os.path.exists(images_output_dir):
        os.makedirs(images_output_dir)

    # Setup wandb if needed
    if wandb_run_name is None:
        args.wandb_run_name = f"SP_{'edge' if edgewise else 'node'}_{case.get_index()}_{args.lambda_reg}{'_zero' if zero_ablation else ''}"

    tl_model.reset_hooks()
    if edgewise:
        masked_model = EdgeLevelMaskedTransformer(tl_model)
    else:
        masked_model = NodeLevelMaskedTransformer(tl_model)
    masked_model = masked_model.to(args.device)

    masked_model.freeze_weights()
    print("Finding subnetwork...")
    if edgewise:
        masked_model, log_dict = train_edge_sp(
            args=args,
            masked_model=masked_model,
            all_task_things=all_task_things,
            print_every=args.print_every,
            eval_fn=partial(eval_fn, ll_model=tl_model, hl_ll_corr=hl_ll_corr),
        )
        percentage_binary = masked_model.proportion_of_binary_scores()
        sp_circuit = build_acdc_circuit(corr=masked_model.get_edge_level_correspondence_from_masks())
    else:
        masked_model, log_dict = train_sp(
            args=args,
            masked_model=masked_model,
            all_task_things=all_task_things,
        )
        from subnetwork_probing.train import proportion_of_binary_scores

        percentage_binary = proportion_of_binary_scores(masked_model)
        corr, _ = iterative_correspondence_from_mask(masked_model.model, log_dict["nodes_to_mask"])
        sp_circuit = build_acdc_circuit(corr=corr)
        print(corr)

    # Update dict with some different things
    # log_dict["nodes_to_mask"] = list(map(str, log_dict["nodes_to_mask"]))
    # to_log_dict["number_of_edges"] = corr.count_no_edges() TODO
    log_dict["percentage_binary"] = percentage_binary

    if calculate_fpr_tpr:
        print("Calculating FPR and TPR for regularizer", args.lambda_reg)
        result = evaluate_acdc_circuit(sp_circuit, tl_model, hl_ll_corr, verbose=False)
    else:
        result = {}

    if calculate_fpr_tpr:
        nodes_fpr = result["nodes"]["fpr"]
        nodes_tpr = result["nodes"]["tpr"]
        edges_fpr = result["edges"]["fpr"]
        edges_tpr = result["edges"]["tpr"]
        if using_wandb:
            wandb.log(
                {
                    "regularizer": args.lambda_reg,
                    "nodes_fpr": nodes_fpr,
                    "nodes_tpr": nodes_tpr,
                    "edges_fpr": edges_fpr,
                    "edges_tpr": edges_tpr,
                }
            )
    if using_wandb:
        wandb.log({"regularizer": args.lambda_reg, "percentage_binary": percentage_binary})
        wandb.finish()
    # print("Done running sp: ")
    # print(result['nodes'])
    return sp_circuit, result
