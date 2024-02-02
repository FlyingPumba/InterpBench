# %%

"""
For now, this is a notebook to train a model with IIT on one of the RASP datasets.

TODO
- define low-level model
- Extend IIT code to support indexes in high-level nodes
- Support for multiple classes
"""

import transformer_lens as tl

from commands.compilation.compile_benchmark import build_tracr_model, build_transformer_lens_model
from benchmark.benchmark_case import BenchmarkCase
import submodules.iit.model_pairs as mp
import submodules.iit.utils.index as index
import correspondence
from utils.get_cases import get_cases
from commands.build_main_parser import build_main_parser


# reload modules
from importlib import reload
reload(correspondence)
# %%

args, _ = build_main_parser().parse_known_args(["compile",
                                                "-i=3",
                                                "-f",])
cases = get_cases(args)
case = cases[0]

tracr_output = build_tracr_model(case, force=True)
hl_model = build_transformer_lens_model(case,
                                        force=args.force,
                                        tracr_output=tracr_output,
                                        device=args.device)
# %%

# this is the graph node -> hl node correspondence
tracr_hl_corr = correspondence.TracrCorrespondence.from_output(tracr_output)

ll_cfg = None
ll_model = tl.HookedTransformer.from_pretrained("gelu-4l")

tracr_ll_corr = {
    ('is_x_3', None): {(0, 'mlp', None)},
    ('frac_prevs_1', None): {(2, 'attn', 0)},
}

def make_hl_ll_corr(tracr_hl_corr, tracr_ll_corr, hook_name_style='tl') -> dict[mp.HLNode, set[mp.LLNode]]:
    def hook_name(loc, style) -> str:
        layer, attn_or_mlp, unit = loc
        if style == 'tl':
            return f"blocks.{layer}.{attn_or_mlp}.{'hook_result' if attn_or_mlp == 'attn' else 'hook_post'}"
        elif style == 'wrapper':
            return f"mod.blocks.{loc}.mod.{attn_or_mlp}.hook_point"
        else:
            raise ValueError(f"Unknown style {style}")
    
    return {
        mp.HLNode(hook_name(hl_loc, hook_name_style), None, hl_loc): {
            mp.LLNode(hook_name(ll_loc, 'tl'), index.Ix[[ll_loc[-1]]]) # TODO support multiple index types?
            for ll_loc in tracr_ll_corr[basis_dir.name, basis_dir.value]
        }
        for basis_dir, hl_loc in tracr_hl_corr.items()
    }

hl_ll_corr = make_hl_ll_corr(tracr_hl_corr, tracr_ll_corr)

model_pair = mp.IITProbeSequentialPair(
    hl_model = hl_model,
    ll_model = ll_model,
    corr = hl_ll_corr
)
# %%
mp.HLNode
# %%
