from argparse import Namespace

import numpy as np
import torch as t

import circuits_benchmark.utils.iit.correspondence as correspondence
import iit.model_pairs as mp
from circuits_benchmark.benchmark.benchmark_case import BenchmarkCase
from circuits_benchmark.commands.common_args import add_common_args
from circuits_benchmark.transformers.hooked_tracr_transformer import HookedTracrTransformer
from circuits_benchmark.utils.iit import make_iit_hl_model
from circuits_benchmark.utils.iit.dataset import get_unique_data, TracrIITDataset, TracrUniqueDataset
from iit.utils.eval_ablations import check_causal_effect, get_causal_effects_for_all_nodes, \
    make_combined_dataframe_of_results, save_result


def setup_args_parser(subparsers):
    parser = subparsers.add_parser("iit")
    add_common_args(parser)

    parser.add_argument("-w", "--weights", type=str, default="510", help="IIT, behavior, strict weights")
    parser.add_argument("-m", "--mean", type=bool, default=True, help="Use mean cache")

    # model_pair_class_map = {
    #     "strict": mp.StrictIITModelPair,
    #     "behavior": mp.IITBehaviorModelPair,
    #     "iit": mp.FreezedModelPair,
    #     "stop_grad": mp.StopGradModelPair
    # }
    # parser.add_argument('-mp', '--model_pair', type=str, default="strict", help="Model pair class to use")


def run_iit_eval(case: BenchmarkCase, args: Namespace):
    output_dir = args.output_dir
    weight = args.weights
    use_mean_cache = args.mean

    if not case.supports_causal_masking():
        raise NotImplementedError(f"Case {case.get_index()} does not support causal masking")

    hl_model = case.build_transformer_lens_model()
    hl_model = make_iit_hl_model(hl_model)
    tracr_output = case.build_tracr_model()
    hl_ll_corr = correspondence.TracrCorrespondence.from_output(case=case, tracr_output=tracr_output)

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
    ll_model.load_weights_from_file(f"{output_dir}/ll_models/{case.get_index()}/ll_model_{weight}.pth")

    model_pair = mp.IITBehaviorModelPair(hl_model, ll_model, hl_ll_corr)

    np.random.seed(0)
    t.manual_seed(0)
    unique_test_data = get_unique_data(case)
    test_set = TracrIITDataset(unique_test_data, unique_test_data, hl_model, every_combination=True)
    result_not_in_circuit = check_causal_effect(model_pair, test_set, node_type="n", verbose=False)
    result_in_circuit = check_causal_effect(model_pair, test_set, node_type="c", verbose=False)

    metric_collection = model_pair._run_eval_epoch(test_set.make_loader(256, 0),
                                                   model_pair.loss_fn)

    # zero/mean ablation
    uni_test_set = TracrUniqueDataset(unique_test_data, unique_test_data, hl_model, every_combination=True)

    za_result_not_in_circuit, za_result_in_circuit = get_causal_effects_for_all_nodes(model_pair, uni_test_set,
                                                                                      batch_size=len(uni_test_set),
                                                                                      use_mean_cache=use_mean_cache)

    df = make_combined_dataframe_of_results(
        result_not_in_circuit,
        result_in_circuit,
        za_result_not_in_circuit,
        za_result_in_circuit,
        use_mean_cache=use_mean_cache)

    save_dir = f"{output_dir}/ll_models/{case.get_index()}/results_{weight}"
    save_result(df, save_dir, model_pair)
    with open(f"{save_dir}/metric_collection.log", "w") as f:
        f.write(str(metric_collection))
        print(metric_collection)