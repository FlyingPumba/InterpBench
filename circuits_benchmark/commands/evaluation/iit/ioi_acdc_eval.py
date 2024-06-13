import os
import pickle
import shutil
from argparse import Namespace
import torch
from circuits_benchmark.transformers.circuit import Circuit
from circuits_benchmark.transformers.circuit_node import CircuitNode
from iit.utils.correspondence import Correspondence
from typing import Optional
from iit.tasks.ioi import ioi_cfg, NAMES, suffixes, make_ll_edges, make_corr_dict
from circuits_benchmark.utils.circuits_comparison import calculate_fpr_and_tpr
from circuits_benchmark.transformers.acdc_circuit_builder import build_acdc_circuit
import transformer_lens
from iit.utils.io_scripts import load_files_from_wandb
from iit.tasks.ioi import make_ioi_dataset_and_hl
from .acdc_utils import ACDCRunner
import iit.model_pairs as mp
from acdc.TLACDCCorrespondence import TLACDCCorrespondence

def setup_args_parser(subparsers):
    parser = subparsers.add_parser("ioi_acdc")
    parser.add_argument(
        "-w",
        "--weights",
        type=str,
        default="100_100_40",
        help="IIT, behavior, strict weights",
    )
    parser.add_argument(
        "--output-dir", type=str, default="./results", help="Output directory"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use", 
    )
    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=0.025,
        help="Threshold for ACDC",
    )
    parser.add_argument("--data-size", type=int, required=False, default=1000, help="How many samples to use")
    parser.add_argument(
        "-wandb", "--using_wandb", action="store_true", help="Use wandb"
    )
    parser.add_argument(
        "--load-from-wandb", action="store_true", help="Load model from wandb"
    )
    parser.add_argument(
        "--include-mlp", action="store_true", help="Evaluate group 'with_mlp'"
    )
    parser.add_argument(
        "--next-token", action="store_true", help="Use next token model"
    )
    parser.add_argument(
        "--use-pos-embed", action="store_true", help="Use positional embeddings"
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


def evaluate_acdc_circuit(
    ll_model: transformer_lens.HookedTransformer,
    acdc_circuit: Circuit,
    corr: Correspondence,
    use_pos_embed: Optional[bool] = False,
    **kwargs,
):
    def make_circuit_node(ll_node: mp.LLNode):
        if 'attn' in ll_node.name:
            index = ll_node.index
            head = index.as_index[2]
            node_name = ll_node.name.replace("hook_z", "hook_result")
            return CircuitNode(node_name, head)
        if "mlp" in ll_node.name:
            node_name = ll_node.name.replace("mlp.hook_post", "hook_mlp_out")
            return CircuitNode(node_name, None)
        return CircuitNode(ll_node.name, None)
        
    edges = make_ll_edges(corr)
    gt_circuit = Circuit()
    for edge in edges:
        circuit_node_from = make_circuit_node(edge[0])
        circuit_node_to = make_circuit_node(edge[1])
        gt_circuit.add_edge(circuit_node_from, circuit_node_to)

    full_corr = TLACDCCorrespondence.setup_from_model(
            ll_model, use_pos_embed=use_pos_embed
    )
    full_circuit = build_acdc_circuit(corr=full_corr)
    return calculate_fpr_and_tpr(
        acdc_circuit, gt_circuit, full_circuit, **kwargs
    )



def run_ioi_acdc(args: Namespace):
    weights = args.weights
    threshold = args.threshold
    using_wandb = args.using_wandb
    device = args.device
    num_samples = args.data_size
    metric = "kl"

    # this is the graph node -> hl node correspondence
    # tracr_hl_corr = correspondence.TracrCorrespondence.from_output(tracr_output)
    output_suffix = f"weight_{weights}/threshold_{threshold}"
    clean_dirname = f"{args.output_dir}/acdc_ioi/{output_suffix}"
    load_dir = os.path.join(
        args.output_dir, "ll_models", f"ioi" if not args.next_token else "ioi_next_token"
    )
    # remove everything in the directory
    if os.path.exists(clean_dirname):
        shutil.rmtree(clean_dirname)

    ll_cfg = transformer_lens.HookedTransformer.from_pretrained("gpt2").cfg.to_dict()
    ll_cfg.update(ioi_cfg)
    
    ll_cfg["use_hook_mlp_in"] = True
    ll_cfg["use_attn_result"] = True
    ll_cfg["use_split_qkv_input"] = True

    ll_model = transformer_lens.HookedTransformer(ll_cfg).to(device)
    if args.load_from_wandb:
        load_files_from_wandb(
            "ioi",
            weights,
            args.next_token,
            [f"ll_model_{weights}.pth", f"corr_{weights}.json"],
            args.output_dir,
            include_mlp=args.include_mlp,
        )
    try:
        ll_model.load_state_dict(
            torch.load(f"{load_dir}/ll_model_{weights}.pth", map_location=device)
        )
    except FileNotFoundError:
        raise FileNotFoundError(f"Model not found at {load_dir}")

    ll_model.eval()
    ll_model.to(device)
    for param in ll_model.parameters():
        param.requires_grad = False

    # make corr
    corr_dict = make_corr_dict(include_mlp=args.include_mlp, eval=True, use_pos_embed=args.use_pos_embed)
    corr = Correspondence.make_corr_from_dict(corr_dict, suffixes)

    # load dataset
    ioi_dataset, hl_model = make_ioi_dataset_and_hl(
        num_samples*2, ll_model, NAMES, verbose=True
    )

    clean_inputs = (ioi_dataset.get_inputs()[:num_samples])
    clean_outputs = ll_model(clean_inputs)

    corrupted_inputs = ioi_dataset.get_inputs()[num_samples:]
    

    label_idx = mp.IOI_ModelPair.get_label_idxs()
    def validation_metric(model_outputs):
        output_slice = model_outputs[label_idx.as_index]
        clean_outputs_slice = clean_outputs[label_idx.as_index]

        return torch.nn.functional.kl_div(
            torch.nn.functional.log_softmax(output_slice, dim=-1),
            torch.nn.functional.softmax(clean_outputs_slice, dim=-1),
            reduction="batchmean",
        )
    print(validation_metric(ll_model(clean_inputs)))
    acdc_runner = ACDCRunner(task="ioi", args=args)
    acdc_circuit, exp = acdc_runner.run_acdc(ll_model, clean_dataset=clean_inputs, corrupt_dataset=corrupted_inputs, validation_metric=validation_metric)

    result = evaluate_acdc_circuit(
        ll_model, acdc_circuit, corr, use_pos_embed=args.use_pos_embed
    )

    # save the result
    with open(f"{clean_dirname}/result.txt", "w") as f:
        f.write(str(result))
    pickle.dump(result, open(f"{clean_dirname}/result.pkl", "wb"))
    print(
        f"Saved result to {clean_dirname}/result.txt and {clean_dirname}/result.pkl"
    )
    if args.using_wandb:
        import wandb
        wandb.init(project=f"circuit_discovery", 
                   group=f"acdc_ioi{'next_token' if args.next_token else ''}_{args.weights}", 
                   name=f"{args.threshold}")
        wandb.save(f"{clean_dirname}/*", base_path=args.output_dir)
    return result
