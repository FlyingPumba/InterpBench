from circuits_benchmark.transformers.hooked_tracr_transformer import HookedTracrTransformer
import torch as t
import numpy as np
from transformer_lens import HookedTransformer
import iit.model_pairs as mp
import iit.utils.index as index
from circuits_benchmark.utils.get_cases import get_cases
from circuits_benchmark.commands.build_main_parser import build_main_parser
from iit_utils import make_iit_hl_model, create_dataset
import iit_utils.correspondence as correspondence
import random
from iit_utils.tracr_ll_corrs import get_tracr_ll_corr
from argparse import Namespace
from argparse import ArgumentParser
import circuits_benchmark.commands.algorithms.acdc as acdc
import os
import shutil

parser = ArgumentParser()
parser.add_argument('-t', '--task', type=int, default=3, help="Task number")
case_num = parser.parse_args().task

args = Namespace(
    command='compile', 
    indices=f'{case_num}', 
    output_dir='/Users/cybershiptrooper/src/interpretability/MATS/circuits-benchmark/results/compile', 
    device='cpu', 
    seed=1234, 
    run_tests=False, 
    tests_atol=1e-05, 
    fail_on_error=False, 
    original_args=['compile', f'-i={case_num}', '-f'])

cases = get_cases(args)
case = cases[0]
if not case.supports_causal_masking():
    raise NotImplementedError(f"Case {case.get_index()} does not support causal masking")

tracr_output = case.build_tracr_model()
hl_model = case.build_transformer_lens_model()
# this is the graph node -> hl node correspondence
tracr_hl_corr = correspondence.TracrCorrespondence.from_output(tracr_output)
clean_dirname = f"results/acdc_{case.get_index()}/"
# remove everything in the directory
if os.path.exists(clean_dirname):
    shutil.rmtree(clean_dirname)
cfg_dict = {
    "n_layers": 2,
    "n_heads": 2,
    "d_head": 4,
    "d_model": 8,
    "d_mlp": 16,
    "seed": 0,
    "act_fn": "gelu",
}
ll_cfg = hl_model.cfg.to_dict().copy()
ll_cfg.update(cfg_dict)

ll_model = HookedTracrTransformer(ll_cfg, hl_model.tracr_input_encoder, hl_model.tracr_output_encoder, hl_model.residual_stream_labels)
# ll_model.load_weights_from_file(f"ll_models/{case_num}/ll_model_110.pth")

print(ll_model.device)
ll_model.to(ll_model.device)
for param in ll_model.parameters():
    print(param.device)
    break

acdc_args, _ = build_main_parser().parse_known_args(['run', 'acdc', '--threshold=0.00', '--metric=l2', ]) #'--data_size=1000'])

acdc.run_acdc(case, acdc_args, ll_model)



# metric_name = "l2"
# validation_metric = case.get_validation_metric(metric_name, ll_model, data_size=1200)
# from iit_utils.dataset import TracrIITDataset, create_dataset, get_encoded_input_from_torch_input
# data, _ = create_dataset(case, hl_model, 1200, 0)
# inputs, outputs, _ = get_encoded_input_from_torch_input(zip(*data.base_data[:]), hl_model, ll_model.device)
# # print(f"Validation metric: {validation_metric}", "\n\noutputs:", outputs)
# print(f"Validation metric: {validation_metric(outputs.unsqueeze(-1))}")

# raise