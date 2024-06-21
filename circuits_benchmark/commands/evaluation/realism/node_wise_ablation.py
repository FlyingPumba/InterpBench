import torch
from transformer_lens import HookedTransformer
from circuits_benchmark.benchmark.benchmark_case import BenchmarkCase
from argparse import Namespace
from circuits_benchmark.commands.common_args import add_common_args
from circuits_benchmark.utils.iit import make_ll_cfg_for_case
from iit.model_pairs.nodes import LLNode
from iit.utils.eval_ablations import get_mean_cache, get_circuit_score
import pickle
import iit.model_pairs as mp
from iit.utils import index
import wandb
from circuits_benchmark.utils.iit.wandb_loader import (
    load_model_from_wandb,
    load_circuit_from_wandb,
)
from circuits_benchmark.transformers.circuit_node import CircuitNode
from circuits_benchmark.transformers.circuit import Circuit


def setup_args_parser(subparsers):
    parser = subparsers.add_parser("node_realism")
    add_common_args(parser)

    parser.add_argument(
        "-w",
        "--weights",
        type=str,
        default="510",
        help="IIT, behavior, strict weights",
    )
    parser.add_argument(
        "-m",
        "--mean",
        action="store_true",
        help="Use mean cache. Defaults to zero ablation if not provided",
    )
    parser.add_argument(
        "--batch_size", type=int, default=512, help="Batch size for evaluation"
    )
    parser.add_argument(
        "-t", "--threshold", type=float, default=0.025, help="Threshold"
    )
    parser.add_argument("--lambda-reg", type=float, default=1.0, help="Regularization")
    parser.add_argument(
        "--use-compressed", action="store_true", help="Use compressed models"
    )
    parser.add_argument("--tracr", action="store_true", help="Use tracr output")
    parser.add_argument("--relative", type=int, default=1, help="Use relative scores")
    parser.add_argument(
        "--algorithm",
        choices=["acdc", "edge_sp", "node_sp", "eap"],
        default="acdc",
        help="Algorithm to use",
    )
    parser.add_argument(
        "-wandb",
        "--use-wandb",
        action="store_true",
        help="Use wandb for logging",
    )
    parser.add_argument(
        "--load-from-wandb", action="store_true", help="Load model from wandb"
    )
    parser.add_argument(
        "--same-size", action="store_true", help="Use same size for ll model"
    )


def make_result_path(case: BenchmarkCase, args: Namespace):
    root_dir = f"./results/{args.algorithm}_{case.get_index()}"
    if not args.use_compressed:
        if args.tracr:
            root_dir += "/weight_tracr"
        else:
            root_dir += f"/weight_{args.weights}"
    if args.algorithm == "acdc":
        return f"{root_dir}/threshold_{args.threshold}/result.pkl"
    elif args.algorithm in ["edge_sp", "node_sp"]:
        return f"{root_dir}/lambda_{args.lambda_reg}/result.pkl"
    raise ValueError(f"Invalid algorithm: {args.algorithm}")


def make_nodes_to_ablate(
    tl_model: HookedTransformer, hypothesis_nodes: list, threshold: float, verbose=False
):
    show = lambda *args, **kwargs: print(*args, **kwargs) if verbose else None
    attn = [
        # LLNode(f"blocks.{layer}.attn.hook_result", index.Ix[:, :, head])
        CircuitNode(f"blocks.{layer}.attn.hook_result", head)
        for layer in range(tl_model.cfg.n_layers)
        for head in range(tl_model.cfg.n_heads)
    ]
    mlps = [
        # LLNode(f"blocks.{layer}.hook_mlp_out", index.Ix[[None]])
        CircuitNode(f"blocks.{layer}.hook_mlp_out", None)
        for layer in range(tl_model.cfg.n_layers)
    ]
    nodes_to_ablate = Circuit()
    for node in attn + mlps:
        nodes_to_ablate.add_node(node)

    for node in hypothesis_nodes:
        if node in nodes_to_ablate:
            show(f"Not ablating node: {node}")
            nodes_to_ablate.remove_node(node)
            assert node.name in tl_model.hook_dict.keys(), ValueError(
                f"{node.name} not in {tl_model.hook_dict.keys()}"
            )
        else:
            show(f"Node {node} not in list")

    ll_nodes_to_ablate = []
    for node in nodes_to_ablate:
        if "attn" in node.name:
            ll_nodes_to_ablate.append(LLNode(node.name, index.Ix[:, :, node.index]))
        else:
            ll_nodes_to_ablate.append(LLNode(node.name, index.Ix[[None]]))
    return ll_nodes_to_ablate


def run_nodewise_ablation(case: BenchmarkCase, args: Namespace):
    output_dir = args.output_dir
    weight = args.weights
    use_mean_cache = args.mean
    use_wandb = args.use_wandb

    hl_model = case.build_transformer_lens_model()
    hl_model = make_iit_hl_model(hl_model, eval_mode=True)

    if args.tracr:
        model = case.get_tl_model()
    else:
        if args.load_from_wandb:
            load_model_from_wandb(
                case.get_index(), weight, output_dir, same_size=args.same_size
            )
        # make ll cfg
        try:
            ll_cfg = pickle.load(
                open(
                    f"{args.output_dir}/ll_models/{case.get_index()}/ll_model_cfg_{weight}.pkl",
                    "rb",
                )
            )
        except FileNotFoundError:
            ll_cfg = make_ll_cfg_for_case(
                hl_model, case.get_index(), same_size=args.same_size
            )
        ll_cfg['device'] = args.device
        model = HookedTransformer(ll_cfg)
        model.load_state_dict(
            torch.load(
                f"{output_dir}/ll_models/{case.get_index()}/ll_model_{weight}.pth",
                map_location=args.device,
            )
        )

    model_pair = mp.IITBehaviorModelPair(
        hl_model=hl_model,
        ll_model=model,
        corr={},
        training_args={},
    )

    unique_test_data = get_unique_data(case, max_len=100_000)
    test_set = TracrUniqueDataset(
        unique_test_data, unique_test_data, hl_model, every_combination=True
    )
    mean_cache = None
    if use_mean_cache:
        mean_cache = get_mean_cache(model_pair, test_set, batch_size=args.batch_size)
    if args.load_from_wandb:
        hyperparam = (
            args.lambda_reg
            if args.algorithm in ["edge_sp", "node_sp"]
            else args.threshold
        )
        result_file = load_circuit_from_wandb(
            case.get_index(),
            args.algorithm,
            hyperparam=str(hyperparam),
            weights=weight,
            output_dir=output_dir,
            same_size=args.same_size,
        )
        result = pickle.load(open(output_dir + "/" + result_file.name, "rb"))
    else:
        result_path = make_result_path(case, args)
        # edges = pickle.load(open(edges_path, "rb"))
        result = pickle.load(open(result_path, "rb"))
    nodes_in_hypothesis = list(
        result["nodes"]["true_positive"] | result["nodes"]["false_positive"]
    )
    nodes_to_ablate = make_nodes_to_ablate(model, nodes_in_hypothesis, args.threshold)
    print("Ablating nodes: ", *nodes_to_ablate, sep="\n")
    score = get_circuit_score(
        model_pair,
        test_set,
        nodes_to_ablate,
        mean_cache,
        use_mean_cache=use_mean_cache,
        batch_size=args.batch_size,
        relative_change=bool(args.relative),
    )

    print(f"Score: {score}")

    if use_wandb:
        group = (
            f"{args.algorithm}_{case.get_index()}_{args.weights}"
            if not args.tracr
            else f"{args.algorithm}_{case.get_index()}_tracr"
        )
        name = (
            f"{args.threshold}"
            if args.algorithm == "acdc"
            else (
                f"{args.lambda_reg}" if args.algorithm in ["edge_sp", "node_sp"] else ""
            )
        )
        wandb.init(
            project=f"node_realism{'_same_size' if args.same_size else ''}",
            group=group,
            name=name,
        )
        wandb.log({"score": score})
