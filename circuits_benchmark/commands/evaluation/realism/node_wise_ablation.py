import torch
from transformer_lens import HookedTransformer
from circuits_benchmark.benchmark.benchmark_case import BenchmarkCase
from argparse import Namespace
from circuits_benchmark.commands.common_args import add_common_args
from circuits_benchmark.utils.iit import make_iit_hl_model
from circuits_benchmark.utils.iit.dataset import (
    get_unique_data,
    TracrIITDataset,
    TracrUniqueDataset,
)
from iit.utils.eval_ablations import get_mean_cache, get_circuit_score
import pickle
import iit.model_pairs as mp
from iit.utils import index


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
    parser.add_argument(
        "--use-compressed", action="store_true", help="Use compressed models"
    )
    parser.add_argument("--tracr", action="store_true", help="Use tracr output")
    parser.add_argument(
        "--relative", type=int, default=1, help="Use relative scores"
    )
    parser.add_argument(
        "--algorithm",
        choices=["acdc", "edge_sp", "node_sp"],
        default="acdc",
        help="Algorithm to use",
    )


def make_edges_path(case: BenchmarkCase, args: Namespace):
    root_dir = f"./results/{args.algorithm}_{case.get_index()}"
    if not args.use_compressed:
        if args.tracr:
            root_dir += "/weight_tracr"
        else:
            root_dir += f"/weight_{args.weights}"
    if args.algorithm == "acdc":
        return f"{root_dir}/threshold_{args.threshold}/edges.pkl"
    return f"{root_dir}/edges.pkl"


def make_nodes_to_ablate(
    tl_model: HookedTransformer, edges: list, threshold: float, verbose=False
):
    show = lambda *args, **kwargs: print(*args, **kwargs) if verbose else None
    attn = [
        mp.LLNode(f"blocks.{layer}.attn.hook_result", index.Ix[:, :, head])
        for layer in range(tl_model.cfg.n_layers)
        for head in range(tl_model.cfg.n_heads)
    ]
    mlps = [
        mp.LLNode(f"blocks.{layer}.hook_mlp_out", index.Ix[[None]])
        for layer in range(tl_model.cfg.n_layers)
    ]
    nodes = attn + mlps
    for edge, score in edges:
        if score is not None and score > threshold:
            from_node = mp.LLNode(edge[0], index.TorchIndex(edge[1].as_index))
            to_node = mp.LLNode(edge[2], index.TorchIndex(edge[3].as_index))
            # find the nodes in the list and remove them
            if from_node in nodes:
                show(f"Not ablating node: {from_node}")
                nodes.remove(from_node)
                assert from_node.name in tl_model.hook_dict.keys(), ValueError(
                    f"{from_node.name} not in {tl_model.hook_dict.keys()}"
                )
            else:
                show(f"Node {from_node} not in list")
            if to_node in nodes:
                show(f"Not ablating node: {to_node}")
                nodes.remove(to_node)
                assert to_node.name in tl_model.hook_dict.keys(), ValueError(
                    f"{to_node.name} not in {tl_model.hook_dict.keys()}"
                )
            else:
                show(f"Node {to_node} not in list")
    return nodes


def run_nodewise_ablation(case: BenchmarkCase, args: Namespace):
    output_dir = args.output_dir
    weight = args.weights
    use_mean_cache = args.mean

    if not case.supports_causal_masking():
        raise NotImplementedError(
            f"Case {case.get_index()} does not support causal masking"
        )

    hl_model = case.build_transformer_lens_model()
    hl_model = make_iit_hl_model(hl_model)
    tracr_output = case.get_tracr_output()

    ll_cfg = hl_model.cfg.to_dict().copy()
    n_heads = max(4, ll_cfg["n_heads"])
    d_head = ll_cfg["d_head"] // 2
    d_model = n_heads * d_head
    d_mlp = d_model * 4
    cfg_dict = {
        "n_layers": max(2, ll_cfg["n_layers"]),
        "n_heads": n_heads,
        "d_head": d_head,
        "d_model": d_model,
        "d_mlp": d_mlp,
        "seed": 0,
        "act_fn": "gelu",
        # "initializer_range": 0.02,
    }
    ll_cfg.update(cfg_dict)

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

    unique_test_data = get_unique_data(case)
    test_set = TracrUniqueDataset(
        unique_test_data, unique_test_data, hl_model, every_combination=True
    )
    mean_cache = None
    if use_mean_cache:
        mean_cache = get_mean_cache(
            model_pair, test_set, batch_size=args.batch_size
        )
    edges_path = make_edges_path(case, args)
    edges = pickle.load(open(edges_path, "rb"))
    nodes_to_ablate = make_nodes_to_ablate(model, edges, args.threshold)
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
