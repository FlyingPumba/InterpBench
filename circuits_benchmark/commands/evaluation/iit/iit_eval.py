import pickle
from argparse import Namespace

import numpy as np
import torch as t
from transformer_lens import HookedTransformer

from circuits_benchmark.benchmark.benchmark_case import BenchmarkCase
from circuits_benchmark.benchmark.tracr_benchmark_case import TracrBenchmarkCase
from circuits_benchmark.benchmark.tracr_dataset import TracrDataset
from circuits_benchmark.commands.common_args import add_common_args
from circuits_benchmark.transformers.hooked_tracr_transformer import (
    HookedTracrTransformer,
)
from circuits_benchmark.utils.iit.best_weights import get_best_weight
from circuits_benchmark.utils.iit.iit_hl_model import IITHLModel
from circuits_benchmark.utils.iit.wandb_loader import load_model_from_wandb
from iit.model_pairs.base_model_pair import BaseModelPair
from iit.utils import IITDataset
from iit.utils.eval_ablations import (
    check_causal_effect,
    get_causal_effects_for_all_nodes,
    make_combined_dataframe_of_results,
    save_result,
    Categorical_Metric,
)


def setup_args_parser(subparsers):
    parser = subparsers.add_parser("iit")
    add_common_args(parser)

    parser.add_argument(
        "-w",
        "--weights",
        type=str,
        default="510",
        help="IIT, behavior, strict weights",
    )
    parser.add_argument("-m", "--mean", type=int, default=1, help="Use mean cache")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Batch size for making mean cache (if using mean ablation)",
    )
    parser.add_argument(
        "--categorical-metric",
        choices=["accuracy", "kl_div", "kl_div_self"],
        default="accuracy",
        help="Categorical metric to use",
    )
    parser.add_argument(
        "--max-len", type=int, default=18000, help="Max length of unique data"
    )
    parser.add_argument(
        "--same-size", action="store_true", help="Use same size for ll model"
    )

    parser.add_argument(
        "--next-token", action="store_true", help="Use next token model"
    )
    parser.add_argument(
        "--include-mlp", action="store_true", help="Evaluate group 'with_mlp'"
    )

    parser.add_argument(
        "--use-wandb", action="store_true", help="Use wandb for logging"
    )
    parser.add_argument(
        "--save-to-wandb", action="store_true", help="Save results to wandb"
    )
    parser.add_argument(
        "--load-from-wandb", action="store_true", help="Load model from wandb"
    )


def get_node_effects(
    case: BenchmarkCase,
    args: Namespace,
    model_pair: BaseModelPair,
    use_mean_cache: bool,
    individual_nodes: bool = True,
):
    np.random.seed(args.seed)
    t.manual_seed(args.seed)

    unique_dataset = case.get_clean_data(max_samples=args.max_len, unique_data=True)
    test_set = IITDataset(unique_dataset, unique_dataset, every_combination=True)

    with t.no_grad():
        result_not_in_circuit = check_causal_effect(
            model_pair,
            test_set,
            node_type="n",
            categorical_metric=Categorical_Metric(args.categorical_metric),
            verbose=False,
        )
        result_in_circuit = check_causal_effect(
            model_pair,
            test_set,
            node_type="c" if not individual_nodes else "individual_c",
            categorical_metric=Categorical_Metric(args.categorical_metric),
            verbose=False,
        )

        metric_collection = model_pair._run_eval_epoch(
            test_set.make_loader(args.batch_size, 0), model_pair.loss_fn
        )

        # zero/mean ablation
        za_result_not_in_circuit, za_result_in_circuit = (
            get_causal_effects_for_all_nodes(
                model_pair,
                unique_dataset,
                batch_size=len(unique_dataset),
                use_mean_cache=use_mean_cache,
            )
        )

    df = make_combined_dataframe_of_results(
        result_not_in_circuit,
        result_in_circuit,
        za_result_not_in_circuit,
        za_result_in_circuit,
        use_mean_cache=use_mean_cache,
    )
    return df, metric_collection


def run_iit_eval(case: BenchmarkCase, args: Namespace):
    output_dir = args.output_dir
    weight = args.weights
    use_mean_cache = args.mean

    hl_model = case.get_hl_model()
    if isinstance(hl_model, HookedTracrTransformer):
        hl_model = IITHLModel(hl_model, eval_mode=True)

    if weight == "tracr":
        assert isinstance(case, TracrBenchmarkCase)
        ll_model = case.get_hl_model()
        hl_ll_corr = case.get_correspondence(same_size=True)
    else:
        if weight == "best":
            weight = get_best_weight(case.get_name())
        
        # make correspondence
        get_correspondence_args = {
            "same_size": args.same_size
        }
        if args.include_mlp:
            get_correspondence_args["include_mlp"] = True
        hl_ll_corr = case.get_correspondence(**get_correspondence_args)

        # load from wandb if needed
        if args.load_from_wandb:
            load_model_from_wandb(
                case.get_name(), weight, output_dir, same_size=args.same_size
            )

        # make ll model
        try:
            ll_cfg = pickle.load(
                open(
                    f"{output_dir}/ll_models/{case.get_name()}/ll_model_cfg_{weight}.pkl",
                    "rb",
                )
            )
        except FileNotFoundError:
            ll_cfg = case.get_ll_model_cfg(same_size=args.same_size)

        ll_cfg["device"] = args.device
        ll_model = HookedTransformer(ll_cfg)
        ll_model.load_state_dict(t.load(
            f"{output_dir}/ll_models/{case.get_name()}/ll_model_{weight}.pth",
            map_location=args.device))
        ll_model.eval()
        ll_model.requires_grad_(False)

    model_pair = case.build_model_pair(hl_model=hl_model, ll_model=ll_model, hl_ll_corr=hl_ll_corr)
    df, metric_collection = get_node_effects(case, args, model_pair, use_mean_cache)

    save_dir = f"{output_dir}/ll_models/{case.get_name()}/results_{weight}"
    suffix = f"_{args.categorical_metric}" if hl_model.is_categorical() else ""
    save_result(df, save_dir, model_pair, suffix=suffix)
    with open(f"{save_dir}/metric_collection.log", "w") as f:
        f.write(str(metric_collection))
        print(metric_collection)

    if args.save_to_wandb:
        import wandb

        wandb.init(
            project=f"node_effect{'_same_size' if args.same_size else ''}",
            tags=[
                f"case_{case.get_name()}",
                f"weight_{weight}",
                f"metric{suffix}",
            ],
            name=f"case_{case.get_name()}_weight_{weight}{suffix}",
        )
        wandb.log(metric_collection.to_dict())
        wandb.save(f"{output_dir}/ll_models/{case.get_name()}/*")
        wandb.save(f"{save_dir}/*")
        wandb.finish()
