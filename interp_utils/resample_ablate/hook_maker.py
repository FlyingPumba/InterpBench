from typing import Callable

import torch
from fancy_einsum import einsum
from torch import Tensor
from transformer_lens.hook_points import HookPoint

from iit.model_pairs.nodes import LLNode
import iit.utils.index as index


def make_scaled_subspace_ablation_hook(
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
        idx = ll_node.index if ll_node.index is not None else index.Ix[[None]]
        cached_activation = model_pair.ll_cache[hook.name][idx.as_index]
        pca_dirs_at_node = pca_dirs[ll_node]
        for i in range(0, cached_activation.shape[1]-1):
            pca_dirs_at_i = pca_dirs_at_node[i]
            components_at_clean_dir = []
            for component in range(pca_dirs_at_i.shape[0]):
                components_at_clean_dir.append(
                    einsum(
                        "batch d_model, d_model -> batch",
                        out[idx.as_index][:, i+1],
                        torch.tensor(pca_dirs_at_i[component]).to(out.device)
                    ).unsqueeze(1)
                )
            
            
            if self_patch and ablate_high_variance:
                # take the pca direction as the direction to remove
                components_to_remove = sum(components_at_clean_dir)
                components_to_add = components_to_remove * scale
            elif self_patch and not ablate_high_variance:
                # take the mean direction as the direction to remove
                components_to_remove = out[idx.as_index][:, i+1] - sum(components_at_clean_dir) 
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
                components_to_remove = out[idx.as_index][:, i+1] - sum(components_at_clean_dir) 
                for component in range(pca_dirs_at_i.shape[0]):
                    components_at_cached_dir.append(
                        einsum(
                            "batch d_model, d_model -> batch",
                            cached_activation[:, i+1],
                            torch.tensor(pca_dirs_at_i[component]).to(out.device),
                        ).unsqueeze(1)
                    )
                components_to_add = (cached_activation[:, i+1] - sum(components_at_cached_dir)) * scale

            out[idx.as_index][:, i+1] = (
                out[idx.as_index][:, i+1] + components_to_add - components_to_remove
            )
            if self_patch and scale == 1:
                # should be the same as the original activations
                assert torch.allclose(out, hook_point_out, atol=1e-5), (out - hook_point_out).abs().max()
        return out

    return ll_ablation_hook

def make_scaled_ablation_hook(
        model_pair, ll_node: LLNode, scale: float
    ) -> Callable[[Tensor, HookPoint], Tensor]:
        """
        Resample ablations, but with the patched activations scaled by the given factor.
        """
        if ll_node.subspace is not None:
            raise NotImplementedError

        def ll_ablation_hook(hook_point_out: Tensor, hook: HookPoint) -> Tensor:
            out = hook_point_out.clone()
            idx = ll_node.index if ll_node.index is not None else index.Ix[[None]]
            out[idx.as_index] = model_pair.ll_cache[hook.name][idx.as_index] * scale
            return out

        return ll_ablation_hook


def make_hook(self_patch, ablate_high_variance, pca_dirs):
    def hook_maker(model_pair, ll_node: LLNode, scale: float) -> Callable[[Tensor, HookPoint], Tensor]:
        return make_scaled_subspace_ablation_hook(model_pair, ll_node, pca_dirs, scale, self_patch=self_patch, ablate_high_variance=ablate_high_variance)
    return hook_maker