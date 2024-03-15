from circuits_benchmark.transformers.hooked_tracr_transformer import HookedTracrTransformer
import iit.model_pairs as mp
import iit.utils.index as index
from iit_utils.dataset import create_dataset, TracrDataset, TracrIITDataset
import iit_utils.correspondence as correspondence
from circuits_benchmark.utils.get_cases import get_cases
from circuits_benchmark.commands.build_main_parser import build_main_parser
from iit_utils.iit_hl_model import make_iit_hl_model
from argparse import Namespace, ArgumentParser
from iit_utils.evals import *
import torch as t
import numpy as np
from iit_utils.dataset import TracrUniqueDataset

# model_pair_class_map = {
#     "strict": mp.StrictIITModelPair,
#     "behavior": mp.IITBehaviorModelPair,
#     "iit": mp.FreezedModelPair,
#     "stop_grad": mp.StopGradModelPair
# }

parser = ArgumentParser()
parser.add_argument("-t", "--task", type=int, default=3, help="Task number")
parser.add_argument("-w", "--weights", type=str, default="510", help="IIT, behavior, strict weights")
parser.add_argument("-m", "--mean", type=bool, default=True, help="Use mean cache")
# parser.add_argument('-mp', '--model_pair', type=str, default="strict", help="Model pair class to use")

_args = parser.parse_args()
case_num = _args.task
weight = _args.weights
use_mean_cache = _args.mean

args = Namespace(
    command="compile",
    indices=f"{case_num}",
    output_dir="results/compile",
    device="cpu",
    seed=1234,
    run_tests=False,
    tests_atol=1e-05,
    fail_on_error=False,
    original_args=["compile", f"-i={case_num}", "-f"],
)
threshold = 0.025

cases = get_cases(args)
case = cases[0]
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
    ll_cfg, hl_model.tracr_input_encoder, hl_model.tracr_output_encoder, hl_model.residual_stream_labels
)
ll_model.load_weights_from_file(f"ll_models/{case_num}/ll_model_{weight}.pth")

model_pair = mp.IITBehaviorModelPair(hl_model, ll_model, hl_ll_corr)

np.random.seed(0)
t.manual_seed(0)
unique_test_data = get_unique_data(case)
test_set = TracrIITDataset(unique_test_data, unique_test_data, hl_model, every_combination=True)
result_not_in_circuit = check_causal_effect(model_pair, test_set, node_type="n", verbose=False)
result_in_circuit = check_causal_effect(model_pair, test_set, node_type="c", verbose=False)

metric_collection = model_pair._run_eval_epoch(test_set.make_loader(256, 0), model_pair.loss_fn)

# zero/mean ablation
uni_test_set = TracrUniqueDataset(unique_test_data, unique_test_data, hl_model, every_combination=True)

za_result_not_in_circuit = check_causal_effect_on_ablation(model_pair, uni_test_set, node_type="n", verbose=False,  use_mean_cache=use_mean_cache)
za_result_in_circuit = check_causal_effect_on_ablation(model_pair, uni_test_set, node_type="c", verbose=False,  use_mean_cache=use_mean_cache)

df = make_combined_dataframe_of_results(result_not_in_circuit, result_in_circuit, za_result_not_in_circuit, za_result_in_circuit, use_mean_cache=use_mean_cache)

save_dir = f"ll_models/{case_num}/results_{weight}"
save_result(df, save_dir, model_pair)
with open(f"{save_dir}/metric_collection.log", "w") as f:
    f.write(str(metric_collection))
    print(metric_collection)