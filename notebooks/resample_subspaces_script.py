# %%
import os
from argparse import ArgumentParser

import torch

import iit.model_pairs as mp
from circuits_benchmark.benchmark.benchmark_case import BenchmarkCase
from circuits_benchmark.utils.get_cases import get_cases
from circuits_benchmark.utils.ll_model_loader.ll_model_loader_factory import (
    get_ll_model_loader,
)
from interp_utils.resample_ablate.collect_cache import (
    collect_activations,
    collect_pca_directions,
)
from interp_utils.resample_ablate.get_ablation_effect import (
    get_ablation_effects_for_scales,
)
from interp_utils.resample_ablate.hook_maker import make_hook, make_scaled_ablation_hook
from interp_utils.resample_ablate.plot_utils import plot_causal_effect

parser = ArgumentParser()
parser.add_argument("--task", type=str, default="3")
parser.add_argument("--max_len", type=int, default=100)
task_idx = parser.parse_args().task_idx
max_len = parser.parse_args().max_len
out_dir = f'./interp_results/{task_idx}/ablate_subspace/'

os.makedirs(out_dir, exist_ok=True)

task: BenchmarkCase = get_cases(indices=[task_idx])[0]

ll_model_loader = get_ll_model_loader(task, interp_bench=True)
hl_ll_corr, model = ll_model_loader.load_ll_model_and_correspondence(device='cuda' if torch.cuda.is_available() else 'cpu')
# turn off grads
model.eval()
model.requires_grad_(False)

hl_model = task.get_hl_model()
model_pair = mp.StrictIITModelPair(hl_model, model, hl_ll_corr)

# %%
unique_test_data = task.get_clean_data(max_samples=max_len, unique_data=True)

loader = torch.utils.data.DataLoader(unique_test_data, batch_size=256, shuffle=False, drop_last=False)

# %%

scales = [0.0, 0.1, 0.2, 0.5, 0.7, 0.8, 1.0, 1.2, 1.4, 2.0]
combined_scales_df = get_ablation_effects_for_scales(model_pair, unique_test_data, 
                                            hook_maker=make_scaled_ablation_hook,
                                            scales=scales)

# %%
combined_scales_df.rename(columns={"scale 0.0_y": "scale 0.0"}, inplace=True)
combined_scales_df = combined_scales_df.sort_values(by=["status"], ascending=False)
combined_scales_df

# %%

plot_causal_effect(combined_scales_df, scales, image_name=f"causal_effect_scale_{task.get_name()}", out_dir=out_dir)

# %%
loader = torch.utils.data.DataLoader(unique_test_data, batch_size=1024, shuffle=False, drop_last=False)

activation_cache = collect_activations(model_pair, loader=loader)
pca_dirs = collect_pca_directions(activation_cache, num_pca_components=2)
activation_cache.keys()

# %%
combined_scales_df_orthogonal = {}


for self_patch in [True, False]:
    for ablate_high_variance in [True, False]:
        hook_maker = make_hook(self_patch, ablate_high_variance)
        combined_scales_df_orthogonal[(self_patch, ablate_high_variance)] = get_ablation_effects_for_scales(
            model_pair, 
            unique_test_data, 
            hook_maker=hook_maker,
            scales=scales)

# %%
for key, df in combined_scales_df_orthogonal.items():
    df = df.sort_values(by=["status"], ascending=False)
    df = df.rename(columns={"scale 0.0_y": "scale 0.0"})
    combined_scales_df_orthogonal[key] = df

plot_causal_effect(combined_scales_df_orthogonal[(True, True)], scales, f"causal_effect_scale_{task.get_name()}_self_patch_ablate_high_variance", out_dir)
plot_causal_effect(combined_scales_df_orthogonal[(True, False)], scales, f"causal_effect_scale_{task.get_name()}_self_patch_ablate_mean", out_dir)
plot_causal_effect(combined_scales_df_orthogonal[(False, True)], scales, f"causal_effect_scale_{task.get_name()}_other_patched_ablate_high_variance", out_dir)
plot_causal_effect(combined_scales_df_orthogonal[(False, False)], scales, f"causal_effect_scale_{task.get_name()}_other_patched_ablate_mean", out_dir)

# %%



