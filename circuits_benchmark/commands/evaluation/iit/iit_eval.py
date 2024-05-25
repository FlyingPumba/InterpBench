from argparse import Namespace

import numpy as np
import torch as t

import circuits_benchmark.utils.iit.correspondence as correspondence
import iit.model_pairs as mp
from circuits_benchmark.benchmark.benchmark_case import BenchmarkCase
from circuits_benchmark.commands.common_args import add_common_args
from circuits_benchmark.transformers.hooked_tracr_transformer import (
    HookedTracrTransformer,
)
from circuits_benchmark.utils.iit import make_iit_hl_model, make_ll_cfg
from circuits_benchmark.utils.iit.dataset import (
    get_unique_data,
    TracrIITDataset,
    TracrUniqueDataset,
)
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
    parser.add_argument(
        "-m", "--mean", type=bool, default=True, help="Use mean cache"
    )
    parser.add_argument(
        "--save-to-wandb", action="store_true", help="Save results to wandb"
    )
    parser.add_argument(
        "--batch_size", type=int, default=256, help="Batch size"
    )
    parser.add_argument(
        "--categorical-metric",
        choices=["accuracy", "kl_div", "kl_div_self"],
        default="accuracy",
        help="Categorical metric to use",
    )
    # parser.add_argument("-o", "--output_dir", type=str, default="./results", help="Output directory")
    # model_pair_class_map = {
    #     "strict": mp.StrictIITModelPair,
    #     "behavior": mp.IITBehaviorModelPair,
    #     "iit": mp.FreezedModelPair,
    #     "stop_grad": mp.StopGradModelPair
    # }
    # parser.add_argument('-mp', '--model_pair', type=str, default="strict", help="Model pair class to use")


def run_iit_eval(case: BenchmarkCase, args: Namespace):
    output_dir = "./results"
    weight = args.weights
    use_mean_cache = args.mean

    hl_model = case.build_transformer_lens_model()
    hl_model = make_iit_hl_model(hl_model)
    tracr_output = case.get_tracr_output()

    if weight == "tracr":
        ll_model = case.get_tl_model()
        hl_ll_corr = correspondence.TracrCorrespondence.make_identity_corr(
            tracr_output=tracr_output
        )
    else:
        hl_ll_corr = correspondence.TracrCorrespondence.from_output(
            case=case, tracr_output=tracr_output
        )
        ll_cfg = make_ll_cfg(hl_model)
        ll_model = HookedTracrTransformer(
            ll_cfg,
            hl_model.tracr_input_encoder,
            hl_model.tracr_output_encoder,
            hl_model.residual_stream_labels,
            remove_extra_tensor_cloning=True,
        )
        ll_model.load_weights_from_file(
            f"{output_dir}/ll_models/{case.get_index()}/ll_model_{weight}.pth"
        )

    model_pair = mp.IITBehaviorModelPair(hl_model, ll_model, hl_ll_corr)

    np.random.seed(0)
    t.manual_seed(0)
    unique_test_data = get_unique_data(case)
    test_set = TracrIITDataset(
        unique_test_data, unique_test_data, hl_model, every_combination=True
    )
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
        node_type="c",
        categorical_metric=Categorical_Metric(args.categorical_metric),
        verbose=False,
    )

    metric_collection = model_pair._run_eval_epoch(
        test_set.make_loader(args.batch_size, 0), model_pair.loss_fn
    )

    # zero/mean ablation
    uni_test_set = TracrUniqueDataset(
        unique_test_data, unique_test_data, hl_model, every_combination=True
    )

    za_result_not_in_circuit, za_result_in_circuit = (
        get_causal_effects_for_all_nodes(
            model_pair,
            uni_test_set,
            batch_size=len(uni_test_set),
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

    save_dir = f"{output_dir}/ll_models/{case.get_index()}/results_{weight}"
    suffix = f"_{args.categorical_metric}" if hl_model.is_categorical() else ""
    save_result(df, save_dir, model_pair, suffix=suffix)
    with open(f"{save_dir}/metric_collection.log", "w") as f:
        f.write(str(metric_collection))
        print(metric_collection)

    if args.save_to_wandb:
        import wandb

        wandb.init(
            project="iit",
            group="eval",
            tags=[
                f"case_{case.get_index()}",
                f"weight_{weight}",
                f"metric{suffix}",
            ],
            name=f"node_effect_{case.get_index()}_weight_{weight}{suffix}",
        )
        wandb.log(metric_collection.to_dict())
        wandb.save(f"{output_dir}/ll_models/{case.get_index()}/*")
        wandb.save(f"{save_dir}/*")
        wandb.finish()
