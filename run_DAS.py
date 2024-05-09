from circuits_benchmark.utils import hf
from circuits_benchmark.utils.get_cases import get_cases
from circuits_benchmark.commands.build_main_parser import build_main_parser
import argparse
from circuits_benchmark.transformers.hooked_tracr_transformer import HookedTracrTransformer

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=int, default=3)
parser.add_argument("--model", type=str, default="510")

parser_args = parser.parse_args()

args, _ = build_main_parser().parse_known_args(
    [
        "compile",
        f"-i={parser_args.task}",
        "-f",
    ]
)

cases = get_cases(args)
case = cases[0]
if not case.supports_causal_masking():
    raise NotImplementedError(f"Case {case.get_index()} does not support causal masking")

tracr_output = case.build_tracr_model()
hl_model = case.build_transformer_lens_model()

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
ll_model.load_weights_from_file(f"ll_models/{case.get_index()}/ll_model_{parser_args.model}.pth")

wrapped_ll_model = hf.make_hf_wrapper_from_tl_model(ll_model)

