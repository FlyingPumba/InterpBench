import datetime
import gc
import os
import random
import shutil
import sys

import numpy as np
import torch
import wandb
from torch.nn import init

from acdc.TLACDCExperiment import TLACDCExperiment
from acdc.acdc_graphics import show
from circuits_benchmark.benchmark.benchmark_case import BenchmarkCase
from circuits_benchmark.commands.analysis.acdc_circuit import calculate_fpr_and_tpr
from circuits_benchmark.commands.common_args import add_common_args
from circuits_benchmark.training.compression.linear_compressed_tracr_transformer import LinearCompressedTracrTransformer
from circuits_benchmark.transformers.acdc_circuit_builder import build_acdc_circuit, get_full_acdc_circuit
from circuits_benchmark.transformers.hooked_tracr_transformer import HookedTracrTransformer
from circuits_benchmark.utils.wandb_artifact_download import download_artifact
from acdc.TLACDCCorrespondence import TLACDCCorrespondence


def setup_args_parser(subparsers):
    parser = subparsers.add_parser("acdc")
    add_common_args(parser)

    parser.add_argument("--threshold", type=float, required=True, help="Value for threshold")
    parser.add_argument(
        "--metric", type=str, required=True, choices=["kl", "l2"], help="Which metric to use for the experiment"
    )
    parser.add_argument("--data-size", type=int, required=False, default=1000, help="How many samples to use")

    parser.add_argument(
        "--wandb-checkpoint-project-name",
        type=str,
        required=False,
        help="A project name to download the checkpoint artifact on which to run the experiment",
    )
    parser.add_argument(
        "--wandb-checkpoint-artifact-name",
        type=str,
        required=False,
        help="A artifact name to download the checkpoint artifact on which to run the experiment",
    )
    parser.add_argument(
        "--wandb-checkpoint-type",
        type=str,
        required=False,
        help="A type to download the checkpoint artifact on which to run the experiment",
    )

    parser.add_argument(
        "--first-cache-cpu",
        type=str,
        required=False,
        default="True",
        help="Value for first_cache_cpu (the old name for the `online_cache`)",
    )
    parser.add_argument(
        "--second-cache-cpu",
        type=str,
        required=False,
        default="True",
        help="Value for second_cache_cpu (the old name for the `corrupted_cache`)",
    )
    parser.add_argument("--zero-ablation", action="store_true", help="Use zero ablation")
    parser.add_argument("--using-wandb", action="store_true", help="Use wandb")
    parser.add_argument(
        "--wandb-entity-name",
        type=str,
        required=False,
        default="remix_school-of-rock",
        help="Value for wandb_entity_name",
    )
    parser.add_argument(
        "--wandb-group-name", type=str, required=False, default="default", help="Value for wandb_group_name"
    )
    parser.add_argument(
        "--wandb-project-name", type=str, required=False, default="acdc", help="Value for wandb_project_name"
    )
    parser.add_argument("--wandb-run-name", type=str, required=False, default=None, help="Value for wandb_run_name")
    parser.add_argument("--wandb-dir", type=str, default="/tmp/wandb")
    parser.add_argument("--wandb-mode", type=str, default="online")
    parser.add_argument("--indices-mode", type=str, default="normal")
    parser.add_argument("--names-mode", type=str, default="normal")
    parser.add_argument("--torch-num-threads", type=int, default=0, help="How many threads to use for torch (0=all)")
    parser.add_argument("--max-num-epochs", type=int, default=100_000)
    parser.add_argument("--single-step", action="store_true", help="Use single step, mostly for testing")
    parser.add_argument(
        "--abs-value-threshold", action="store_true", help="Use the absolute value of the result to check threshold"
    )


def run_acdc(
    case: BenchmarkCase,
    args,
    model: HookedTracrTransformer = None,
    calculate_fpr_tpr: bool = False,
    output_suffix: str = "",
):
    if model is None:
        tl_model = case.get_tl_model(device=args.device, remove_extra_tensor_cloning=False)
    else:
        tl_model = model

    tags = [f"case{case.get_index()}", "acdc"]
    notes = f"Command: {' '.join(sys.argv)}"

    if (
        args.wandb_checkpoint_project_name is not None
        or args.wandb_checkpoint_artifact_name is not None
        or args.wandb_checkpoint_type is not None
    ):
        downloaded_files = download_artifact(args.wandb_checkpoint_project_name, args.wandb_checkpoint_artifact_name)

        if len(downloaded_files) == 0:
            raise ValueError(
                f"Failed to download artifact {args.wandb_checkpoint_artifact_name} from project {args.wandb_checkpoint_project_name}"
            )

        weights_file = next((f for f in downloaded_files if f.name.endswith("weights.pt")), None)
        if weights_file is None:
            raise ValueError(f"Failed to find weights file in {downloaded_files}")

        # parse the case index and resid size out of the filename
        case_index = weights_file.name.split("-")[1]
        compression_size = int(weights_file.name.split("-")[3].split(".")[0])

        if case_index != case.get_index():
            raise ValueError(
                f"Case index {case_index} in weights artifact does not match the case index {case.get_index()}"
            )

        tags.append(args.wandb_checkpoint_type)
        if (
            args.wandb_checkpoint_type == "natural-compression"
            or args.wandb_checkpoint_type == "non-linear-compression"
        ):
            tl_model = HookedTracrTransformer.from_hooked_tracr_transformer(
                tl_model,
                overwrite_cfg_dict={"d_model": compression_size},
                init_params_fn=lambda x: init.kaiming_uniform_(x) if len(x.shape) > 1 else init.normal_(x, std=0.02),
                remove_extra_tensor_cloning=False,
            )
            tl_model.load_state_dict(torch.load(weights_file))
        elif args.wandb_checkpoint_type == "linear-compression":
            tl_model = LinearCompressedTracrTransformer(
                tl_model, int(compression_size), "linear", remove_extra_tensor_cloning=False
            )
            tl_model.load_state_dict(torch.load(weights_file))
            tl_model = tl_model.get_folded_model()
        else:
            raise ValueError(f"Unknown wandb_checkpoint_type {args.wandb_checkpoint_type}")

    # Check that dot program is in path
    if not shutil.which("dot"):
        raise ValueError("dot program not in path, cannot generate graphs for ACDC.")

    if args.torch_num_threads > 0:
        torch.set_num_threads(args.torch_num_threads)

    # Set the seed
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.first_cache_cpu is None:
        online_cache_cpu = True
    elif args.first_cache_cpu.lower() == "false":
        online_cache_cpu = False
    elif args.first_cache_cpu.lower() == "true":
        online_cache_cpu = True
    else:
        raise ValueError(f"first_cache_cpu must be either True or False, got {args.first_cache_cpu}")

    if args.second_cache_cpu is None:
        corrupted_cache_cpu = True
    elif args.second_cache_cpu.lower() == "false":
        corrupted_cache_cpu = False
    elif args.second_cache_cpu.lower() == "true":
        corrupted_cache_cpu = True
    else:
        raise ValueError(f"second_cache_cpu must be either True or False, got {args.second_cache_cpu}")

    threshold = args.threshold  # only used if >= 0.0
    metric_name = args.metric
    zero_ablation = True if args.zero_ablation else False
    using_wandb = True if args.using_wandb else False
    wandb_entity_name = args.wandb_entity_name
    wandb_project_name = args.wandb_project_name
    wandb_run_name = args.wandb_run_name
    wandb_group_name = args.wandb_group_name
    indices_mode = args.indices_mode
    names_mode = args.names_mode
    device = args.device
    single_step = True if args.single_step else False

    second_metric = None  # some tasks only have one metric
    use_pos_embed = True  # Always true for all tracr models.

    data_size = args.data_size
    validation_metric = case.get_validation_metric(metric_name, tl_model, data_size=data_size)
    toks_int_values = case.get_clean_data(count=data_size).get_inputs()
    toks_int_values_other = case.get_corrupted_data(count=data_size).get_inputs()

    tl_model.reset_hooks()

    # Create the output directory
    output_dir = os.path.join(args.output_dir, f"acdc_{case.get_index()}", output_suffix)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    images_output_dir = os.path.join(output_dir, f"acdc_{case.get_index()}", "images")
    if not os.path.exists(images_output_dir):
        os.makedirs(images_output_dir)

    # Save some mem
    gc.collect()
    torch.cuda.empty_cache()

    # Setup wandb if needed
    if wandb_run_name is None:
        wandb_run_name = (
            f"{'_randomindices' if indices_mode == 'random' else ''}_{threshold}{'_zero' if zero_ablation else ''}"
        )
    else:
        assert wandb_run_name is not None, "I want named runs, always"

    tl_model.reset_hooks()
    exp = TLACDCExperiment(
        model=tl_model,
        threshold=threshold,
        images_output_dir=images_output_dir,
        using_wandb=using_wandb,
        wandb_entity_name=wandb_entity_name,
        wandb_project_name=wandb_project_name,
        wandb_run_name=wandb_run_name,
        wandb_group_name=wandb_group_name,
        wandb_notes=notes,
        wandb_tags=tags,
        wandb_dir=args.wandb_dir,
        wandb_mode=args.wandb_mode,
        wandb_config=args,
        zero_ablation=zero_ablation,
        abs_value_threshold=args.abs_value_threshold,
        ds=toks_int_values,
        ref_ds=toks_int_values_other,
        metric=validation_metric,
        second_metric=second_metric,
        verbose=True,
        indices_mode=indices_mode,
        names_mode=names_mode,
        corrupted_cache_cpu=corrupted_cache_cpu,
        hook_verbose=False,
        online_cache_cpu=online_cache_cpu,
        add_sender_hooks=True,
        use_pos_embed=use_pos_embed,
        add_receiver_hooks=False,
        remove_redundant=False,
        show_full_index=use_pos_embed,
    )

    exp_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    for i in range(args.max_num_epochs):
        exp.step(testing=False)

        show(
            exp.corr,
            fname=f"{images_output_dir}/img_new_{i + 1}.png",
        )

        print(i, "-" * 50)
        print(exp.count_num_edges())

        if i == 0:
            exp.save_edges(os.path.join(output_dir, "edges.pkl"))

        if exp.current_node is None or single_step:
            show(
                exp.corr,
                fname=f"{images_output_dir}/ACDC_new_{exp_time}.png",
                show_placeholders=True,
            )
            break

    exp.save_edges(os.path.join(output_dir, "another_final_edges.pkl"))

    exp.save_subgraph(
        fpath=f"{output_dir}/subgraph.pth",
        return_it=True,
    )

    acdc_circuit = build_acdc_circuit(exp.corr)
    acdc_circuit.save(f"{output_dir}/final_circuit.pkl")

    if calculate_fpr_tpr:
        print("Calculating FPR and TPR for threshold", threshold)
        full_corr = TLACDCCorrespondence.setup_from_model(tl_model, use_pos_embed=use_pos_embed)
        full_circuit = build_acdc_circuit(full_corr)
        tracr_hl_circuit, tracr_ll_circuit, alignment = case.get_tracr_circuit(granularity="acdc_hooks")
        result = calculate_fpr_and_tpr(acdc_circuit, tracr_ll_circuit, full_circuit, verbose=True)
    else:
        result = {}

    result["current_metric"] = exp.cur_metric

    if using_wandb:
        edges_fname = f"edges.pth"
        exp.save_edges(edges_fname)

        artifact = wandb.Artifact(edges_fname, type="dataset")
        artifact.add_file(edges_fname)
        wandb.log_artifact(artifact)
        if calculate_fpr_tpr:
            nodes_fpr = result["nodes"]["fpr"]
            nodes_tpr = result["nodes"]["tpr"]
            edges_fpr = result["edges"]["fpr"]
            edges_tpr = result["edges"]["tpr"]
            wandb.log(
                {
                    "threshold": threshold,
                    "nodes_fpr": nodes_fpr,
                    "nodes_tpr": nodes_tpr,
                    "edges_fpr": edges_fpr,
                    "edges_tpr": edges_tpr,
                }
            )
        else:
            wandb.log({"threshold": threshold})

        os.remove(edges_fname)
        wandb.finish()

    return acdc_circuit, result
