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

ll_model = HookedTracrTransformer(ll_cfg, hl_model.tracr_input_encoder, hl_model.tracr_output_encoder, hl_model.residual_stream_labels)
ll_model.load_weights_from_file(f"ll_models/{case_num}/ll_model_110.pth")


acdc_args, _ = build_main_parser().parse_known_args(['run', 'acdc', '--threshold=0.71', '--metric=l2'])

acdc.run_acdc(case, acdc_args, ll_model)
