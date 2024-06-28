# %% [markdown]
# Setup

# %%
import pickle
import torch
from transformer_lens import HookedTransformerConfig, HookedTransformer
from transformer_lens import HookedTransformer
from circuits_benchmark.utils.get_cases import get_cases
import os

task = get_cases(indices=['11'])[0]
task_idx = task.get_index()
# create directory for images
os.makedirs(f"resample_ablate_results/case_{task.get_index()}", exist_ok=True)

# %%
dir_name = f"../InterpBench/{task_idx}"
cfg_dict = pickle.load(open(f"{dir_name}/ll_model_cfg.pkl", "rb"))
cfg = HookedTransformerConfig.from_dict(cfg_dict)
cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
model = HookedTransformer(cfg)
weights = torch.load(f"{dir_name}/ll_model.pth", map_location=cfg.device)
model.load_state_dict(weights)
# turn off grads
model.eval()
model.requires_grad_(False)
torch.set_grad_enabled(False)

# %%
# load high level model
from circuits_benchmark.utils.iit import make_iit_hl_model
import circuits_benchmark.utils.iit.correspondence as correspondence
import iit.model_pairs as mp

def make_model_pair(benchmark_case):
    hl_model = benchmark_case.build_transformer_lens_model()
    hl_model = make_iit_hl_model(hl_model, eval_mode=True)
    tracr_output = benchmark_case.get_tracr_output()
    hl_ll_corr = correspondence.TracrCorrespondence.from_output(
            case=benchmark_case, tracr_output=tracr_output
        )
    model_pair = mp.StrictIITModelPair(hl_model, model, hl_ll_corr)
    return model_pair

# %%
from circuits_benchmark.utils.iit.dataset import get_unique_data

max_len = 200
model_pair = make_model_pair(task)
unique_test_data = get_unique_data(task, max_len=max_len)

# %% [markdown]
# Resample ablate with 10%, 20% etc. of the activation

# %%
from iit.model_pairs.nodes import LLNode
from typing import Callable
from torch import Tensor
from transformer_lens.hook_points import HookPoint
import iit.utils.eval_ablations as eval_ablations
from importlib import reload
from circuits_benchmark.utils.iit.dataset import TracrIITDataset
import pandas as pd


def get_effects_for_scales(
    model_pair,
    unique_test_data,
    hook_maker: Callable[
        [mp.BaseModelPair, LLNode, float], Callable[[Tensor, HookPoint], Tensor]
    ],
    scales=[0.1, 1.0],
):
    combined_scales_df = pd.DataFrame(
        columns=["node", "status"] + [f"scale {scale}" for scale in scales]
    )

    for scale in scales:
        print(f"Running scale {scale}\n")
        test_set = TracrIITDataset(
            unique_test_data,
            unique_test_data,
            model_pair.hl_model,
            every_combination=True,
        )

        hook_maker_for_node = lambda ll_node: hook_maker(model_pair=model_pair, ll_node=ll_node, scale=scale)

        causal_effects_not_in_circuit = eval_ablations.check_causal_effect(
            model_pair=model_pair,
            dataset=test_set,
            hook_maker=hook_maker_for_node,
            node_type="n",
            batch_size=1024,
        )

        causal_effects_in_circuit = eval_ablations.check_causal_effect(
            model_pair=model_pair,
            dataset=test_set,
            hook_maker=hook_maker_for_node,
            node_type="individual_c",
            batch_size=1024,
        )

        causal_effects = eval_ablations.make_dataframe_of_results(
            causal_effects_not_in_circuit, causal_effects_in_circuit
        )

        # change column name causal effect to scale
        causal_effects.rename(columns={"causal effect": f"scale {scale}"}, inplace=True)
        combined_scales_df = pd.merge(
            combined_scales_df, causal_effects, on=["node", "status"], how="outer"
        )
        # drop columns with nan
        combined_scales_df.dropna(axis=1, how="all", inplace=True)
    return combined_scales_df

# %%
def make_ll_ablation_hook_scale_activations(
        model_pair, ll_node: LLNode, scale: float
    ) -> Callable[[Tensor, HookPoint], Tensor]:
        """
        Resample ablations, but with the patched activations scaled by the given factor.
        """
        if ll_node.subspace is not None:
            raise NotImplementedError

        def ll_ablation_hook(hook_point_out: Tensor, hook: HookPoint) -> Tensor:
            out = hook_point_out.clone()
            index = ll_node.index if ll_node.index is not None else index.Ix[[None]]
            out[index.as_index] = model_pair.ll_cache[hook.name][index.as_index] * scale
            return out

        return ll_ablation_hook

scales = [0.0, 0.1, 0.2, 0.5, 0.7, 0.8, 1.0, 1.2, 1.4, 2.0, 5.0]
combined_scales_df = get_effects_for_scales(model_pair, unique_test_data, 
                                            hook_maker=make_ll_ablation_hook_scale_activations,
                                            scales=scales)

# # %%
combined_scales_df.rename(columns={"scale 0.0_y": "scale 0.0"}, inplace=True)
combined_scales_df = combined_scales_df.sort_values(by=["status"], ascending=False)
combined_scales_df

# %%
import plotly.graph_objects as go

def plot_causal_effect(combined_scales_df, scales, image_name):
    fig = go.Figure()
    scale_columns = [f"scale {scale}" for scale in scales]
    for i, row in combined_scales_df.iterrows():
        y = [row[col] for col in scale_columns]
        x = scales
        # plot lines for each node
        fig.add_trace(go.Line(x=x, y=y, mode='lines+markers', 
                            # set color based on status
                            line=dict(color="green" if row["status"] == "in_circuit" else "orange"),
                            hovertext=f"Node: {row['node']}, Status: {row['status']}",
                            # define legend only for color, not for line
                            showlegend=False,
                            ),
                    )
    fig.update_layout(xaxis_title="Scale", yaxis_title="Causal Effect")
    # make legend for color
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(color="green"), name="in_circuit"))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(color="orange"), name="not_in_circuit"))
    # make background transparent and remove grid
    fig.update_layout(template="plotly_white")
    # remove grid
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    # decrease margin
    fig.update_layout(margin=dict(l=70, r=70, t=70, b=70))
    # increase font size
    fig.update_layout(font=dict(size=16))
    # add title
    fig.update_layout(title=image_name)
    fig.show()
    # save to file as pdf with same width and height
    fig.write_image(f"resample_ablate_results/case_{task.get_index()}/{image_name}.png")

# %%
plot_causal_effect(combined_scales_df, scales, f"causal_effect_scale_{task.get_index()}")

# %% [markdown]
# Do PCA and patch only the varying part 

# %%
from iit.utils.node_picker import get_nodes_in_circuit, get_nodes_not_in_circuit, get_all_nodes
from sklearn.decomposition import PCA
import iit.utils.index as index

def find_ll_node_by_name(name, list_of_nodes) -> list:
    ll_nodes = []
    for node in list_of_nodes:
        if node.name == name:
            ll_nodes.append(node)
    return ll_nodes


def collect_activations(model_pair, loader, pos_slice=slice(1, None, None)):
    activation_cache = {}
    nodes = get_all_nodes(model_pair.ll_model)
    pos_idx = index.TorchIndex(((slice(None), pos_slice))) if pos_slice is not None else index.Ix[slice(None)]
    for node in nodes:
        activation_cache[node] = None
    for batch in loader:
        _, batch_cache = model_pair.ll_model.run_with_cache(batch)
        for k, tensor in batch_cache.items():
            ll_node_for_k = find_ll_node_by_name(k, nodes)
            if len(ll_node_for_k) > 0:
                for node in ll_node_for_k:
                    act = tensor[node.index.as_index].cpu()[pos_idx.as_index]
                    if activation_cache[node] is None:
                        activation_cache[node] = act
                    else:
                        activation_cache[node] = torch.cat((activation_cache[node], act), dim=0).cpu()
    return activation_cache


def collect_pca_directions(activation_cache, num_pca_components=2):
    pca_dirs = {}

    for node, activations in activation_cache.items():
        # calculate pca directions for activations
        for i in range(activations.shape[1]):
            pca = PCA(n_components=num_pca_components)
            activations_to_pca = activations[:, i].detach().numpy()
            # center data before pca
            activations_to_pca = activations_to_pca - activations_to_pca.mean(axis=0)
            pca.fit(activations_to_pca)
            if pca_dirs.get(node) is None:
                pca_dirs[node] = {}
            pca_dirs[node][i] = pca.components_
    return pca_dirs

# %%
def collate_fn(batch):
    encoded_x = model_pair.hl_model.map_tracr_input_to_tl_input(list(zip(*batch))[0])
    return encoded_x

loader = torch.utils.data.DataLoader(unique_test_data, batch_size=1024, shuffle=False, drop_last=False, collate_fn=collate_fn)

activation_cache = collect_activations(model_pair, loader=loader)
activation_cache.keys()

# %%
pca_dirs = collect_pca_directions(activation_cache, num_pca_components=2)

# %%
from fancy_einsum import einsum


def make_ll_ablation_hook_scale_activations_with_variance(
    model_pair, ll_node: LLNode, pca_dirs: dict, scale: float, 
    self_patch: bool = False,
    ablate_high_variance: bool = True
) -> Callable[[Tensor, HookPoint], Tensor]:
    """
    Resample ablations, but with the patched activations scaled by the given factor, along the PCA directions. Since the PCA directions capture the variance in activations, this may help us to distinguish between constant nodes and nodes whose variance is important for the model. My hypothesis is that constant nodes will have a smaller effect on the model than nodes whose variance is important for any scale provided.

    If self_patch is True, the ablation will be done with the activations at the node itself, rather than the activations at the node in the cache. So this is not a resample ablation.  
    """
    if ll_node.subspace is not None:
        raise NotImplementedError

    def ll_ablation_hook(hook_point_out: Tensor, hook: HookPoint) -> Tensor:
        out = hook_point_out.clone()
        index = ll_node.index if ll_node.index is not None else index.Ix[[None]]
        cached_activation = model_pair.ll_cache[hook.name][index.as_index]
        pca_dirs_at_node = pca_dirs[ll_node]
        for i in range(0, cached_activation.shape[1]-1):
            pca_dirs_at_i = pca_dirs_at_node[i]
            components_at_clean_dir = []
            for component in range(pca_dirs_at_i.shape[0]):
                components_at_clean_dir.append(
                    einsum(
                        "batch d_model, d_model -> batch",
                        out[index.as_index][:, i+1],
                        torch.tensor(pca_dirs_at_i[component]).to(out.device)
                    ).unsqueeze(1)
                )
            
            
            if self_patch and ablate_high_variance:
                # take the pca direction as the direction to remove
                components_to_remove = sum(components_at_clean_dir)
                components_to_add = components_to_remove * scale
            elif self_patch and not ablate_high_variance:
                # take the mean direction as the direction to remove
                components_to_remove = out[index.as_index][:, i+1] - sum(components_at_clean_dir) 
                components_to_add = components_to_remove * scale
            elif not self_patch and ablate_high_variance:
                components_at_cached_dir = []
                components_to_remove = sum(components_at_clean_dir)
                for component in range(pca_dirs_at_i.shape[0]):
                    components_at_cached_dir.append(
                        einsum(
                            "batch d_model, d_model -> batch",
                            cached_activation[:, i+1],
                            torch.tensor(pca_dirs_at_i[component]).to(out.device),
                        ).unsqueeze(1)
                    )
                components_to_add = sum(components_at_cached_dir) * scale
            elif not self_patch and not ablate_high_variance:
                components_at_cached_dir = []
                components_to_remove = out[index.as_index][:, i+1] - sum(components_at_clean_dir) 
                for component in range(pca_dirs_at_i.shape[0]):
                    components_at_cached_dir.append(
                        einsum(
                            "batch d_model, d_model -> batch",
                            cached_activation[:, i+1],
                            torch.tensor(pca_dirs_at_i[component]).to(out.device),
                        ).unsqueeze(1)
                    )
                components_to_add = (cached_activation[:, i+1] - sum(components_at_cached_dir)) * scale

            out[index.as_index][:, i+1] = (
                out[index.as_index][:, i+1] + components_to_add - components_to_remove
            )
            if self_patch and scale == 1:
                # should be the same as the original activations
                assert torch.allclose(out, hook_point_out, atol=1e-5), (out - hook_point_out).abs().max()
        return out

    return ll_ablation_hook

# %%
# %%capture
combined_scales_df_orthogonal = {}
def make_hook(self_patch, ablate_high_variance):
    def hook_maker(model_pair, ll_node: LLNode, scale: float) -> Callable[[Tensor, HookPoint], Tensor]:
        return make_ll_ablation_hook_scale_activations_with_variance(model_pair, ll_node, pca_dirs, scale, self_patch=self_patch, ablate_high_variance=ablate_high_variance)
    return hook_maker

for self_patch in [True, False]:
    for ablate_high_variance in [True, False]:
        hook_maker = make_hook(self_patch, ablate_high_variance)
        combined_scales_df_orthogonal[(self_patch, ablate_high_variance)] = get_effects_for_scales(model_pair, unique_test_data, 
                                            hook_maker=hook_maker,
                                            scales=scales)

# %%
for key, df in combined_scales_df_orthogonal.items():
    df = df.sort_values(by=["status"], ascending=False)
    df = df.rename(columns={"scale 0.0_y": "scale 0.0"})
    combined_scales_df_orthogonal[key] = df

plot_causal_effect(combined_scales_df_orthogonal[(True, True)], scales, f"causal_effect_scale_{task.get_index()}_self_patch_ablate_high_variance")
plot_causal_effect(combined_scales_df_orthogonal[(True, False)], scales, f"causal_effect_scale_{task.get_index()}_self_patch_ablate_mean")
plot_causal_effect(combined_scales_df_orthogonal[(False, True)], scales, f"causal_effect_scale_{task.get_index()}_other_patched_ablate_high_variance")
plot_causal_effect(combined_scales_df_orthogonal[(False, False)], scales, f"causal_effect_scale_{task.get_index()}_other_patched_ablate_mean")

# %% [markdown]
# Check if the PCA dir is orthogonal using logit lens

# %%



