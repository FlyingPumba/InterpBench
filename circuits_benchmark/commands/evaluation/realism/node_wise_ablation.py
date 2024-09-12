import pickle
from argparse import Namespace

import wandb
from iit.model_pairs.iit_behavior_model_pair import IITBehaviorModelPair
from iit.utils import IITDataset, index
from iit.utils.eval_ablations import get_circuit_score, get_mean_cache
from iit.utils.nodes import LLNode
from transformer_lens import HookedTransformer

from circuits_benchmark.benchmark.benchmark_case import BenchmarkCase
from circuits_benchmark.commands.common_args import add_common_args, add_evaluation_common_ags
from circuits_benchmark.transformers.hooked_tracr_transformer import HookedTracrTransformer
from circuits_benchmark.utils.circuit.circuit import Circuit
from circuits_benchmark.utils.circuit.circuit_eval import CircuitEvalResult
from circuits_benchmark.utils.circuit.circuit_node import CircuitNode
from circuits_benchmark.utils.iit.iit_hl_model import IITHLModel
from circuits_benchmark.utils.iit.wandb_loader import load_circuit_from_wandb
from circuits_benchmark.utils.ll_model_loader.ll_model_loader_factory import LLModelLoader, \
    get_ll_model_loader_from_args


def setup_args_parser(subparsers):
    parser = subparsers.add_parser("node_realism")
    add_common_args(parser)
    add_evaluation_common_ags(parser)

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
        "--use-iit-model", action="store_true", help="Use IIT model instead of SIIT model"
    )


def make_result_path(case: BenchmarkCase, args: Namespace, ll_model_loader: LLModelLoader):
    root_dir = f"./results/{args.algorithm}/{case.get_name()}/{ll_model_loader.get_output_suffix()}"
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
    use_mean_cache = args.mean
    use_wandb = args.use_wandb

    hl_model = case.get_hl_model()
    if isinstance(hl_model, HookedTracrTransformer):
        hl_model = IITHLModel(hl_model, eval_mode=True)

    ll_model_loader = get_ll_model_loader_from_args(case, args)
    _, ll_model = ll_model_loader.load_ll_model_and_correspondence(args.device, output_dir=output_dir, same_size=args.same_size)
    ll_model.eval()
    ll_model.requires_grad_(False)

    model_pair = IITBehaviorModelPair(
        hl_model=hl_model,
        ll_model=ll_model,
        corr={},
        training_args={},
    )

    unique_dataset = case.get_clean_data(max_samples=100_000, unique_data=True)
    test_set = IITDataset(unique_dataset, unique_dataset, every_combination=True)
    mean_cache = None
    if use_mean_cache:
        mean_cache = get_mean_cache(model_pair, unique_dataset, batch_size=args.batch_size)
    if args.load_from_wandb:
        hyperparam = (
            args.lambda_reg
            if args.algorithm in ["edge_sp", "node_sp"]
            else args.threshold
        )
        result_file = load_circuit_from_wandb(
            case.get_name(),
            args.algorithm,
            hyperparam=str(hyperparam),
            weights=ll_model_loader.get_output_suffix(),
            output_dir=output_dir,
            same_size=args.same_size,
        )
        result = pickle.load(open(output_dir + "/" + result_file.name, "rb"))
    else:
        result_path = make_result_path(case, args, ll_model_loader)
        print(f"Loading result from {result_path}")
        result: CircuitEvalResult = pickle.load(open(result_path, "rb"))
    nodes_in_hypothesis = list(
        result.nodes.true_positive | result.nodes.false_positive
    )
    nodes_to_ablate = make_nodes_to_ablate(ll_model, nodes_in_hypothesis, args.threshold)
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
        group = f"{args.algorithm}_{case.get_name()}_{ll_model_loader.get_output_suffix()}"
        name = (
            f"{args.threshold}"
            if args.algorithm == "acdc"
            else (
                f"{args.lambda_reg}" if args.algorithm in ["edge_sp", "node_sp"] else ""
            )
        )
        project = "node_realism_same_size" if args.same_size else "node_realism"
        
        wandb.init(
            project=project,
            group=group,
            name=name,
        )
        wandb.log({"score": score})
