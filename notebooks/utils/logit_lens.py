from jaxtyping import Float
from transformer_lens import ActivationCache
from fancy_einsum import einsum
import iit.utils.index as index
import torch
import iit.model_pairs as mp
from dataclasses import dataclass


def residual_stack_to_logit_diff(
    residual_stack: Float[torch.Tensor, "components batch pos d_model"],
    cache: ActivationCache,
    logit_diff_directions: Float[torch.Tensor, "batch pos d_model"],
    pos_slice: slice = slice(1, None, None),
) -> float:
    scaled_residual_stack = cache.apply_ln_to_stack(
        residual_stack, layer=-1, pos_slice=pos_slice
    )
    return einsum(
        "... batch pos d_model, batch pos d_model -> ... batch pos",
        scaled_residual_stack,
        logit_diff_directions,
    )


def do_logit_lens(model_pair: mp.BaseModelPair, loader: torch.utils.data.DataLoader):
    model = model_pair.ll_model
    logit_lens_results = {}
    labels = None
    for batch in loader:
        original_logits, cache = model.run_with_cache(batch)
        pos_slice = slice(1, None, None)
        pos_idx = index.Ix[:, 1:]
        # get residual stack for each layer and head
        per_layer_residual, layers = cache.decompose_resid(
            -1, mode="mlp", return_labels=True, pos_slice=pos_slice
        )
        per_head_residual, attns = cache.stack_head_results(
            layer=-1, pos_slice=pos_slice, return_labels=True
        )

        if model_pair.hl_model.is_categorical():
            control_logit = model.unembed.W_U[:, 0]
            answers_idxs = original_logits.argmax(dim=-1)[pos_idx.as_index]
            logit_diff_directions = model.unembed.W_U.T[answers_idxs] - control_logit
        else:
            # logit diff directions: for regression, we only have one vector in unembed...
            logit_diff_direction = model.unembed.W_U.t().squeeze(0)
            # expand to batch, pos, d_model
            logit_diff_directions = (
                logit_diff_direction.unsqueeze(0)
                .unsqueeze(1)
                .expand(per_layer_residual.shape[1], per_layer_residual.shape[2], -1)
            )
            print(logit_diff_directions.shape, per_layer_residual.shape)

        # logit lens
        per_layer_logit_diff = residual_stack_to_logit_diff(
            per_layer_residual, cache, logit_diff_directions
        )
        per_head_logit_diff = residual_stack_to_logit_diff(
            per_head_residual, cache, logit_diff_directions
        )

        # store results
        for layer, logit_diff in zip(layers, per_layer_logit_diff):
            if layer not in logit_lens_results:
                logit_lens_results[layer] = logit_diff
            else:
                # stack the results at dim 0
                logit_lens_results[layer] = torch.cat(
                    [logit_lens_results[layer], logit_diff], dim=0
                )

        for attn, logit_diff in zip(attns, per_head_logit_diff):
            if attn not in logit_lens_results:
                logit_lens_results[attn] = logit_diff
            else:
                logit_lens_results[attn] = torch.cat(
                    [logit_lens_results[attn], logit_diff], dim=0
                )
        if model_pair.hl_model.is_categorical():
            original_sliced = original_logits[pos_idx.as_index]  # batch, pos, d_model
            control_logit = original_sliced[:, :, 0]
            answers_idxs = original_sliced.argmax(dim=-1)
            # get the logits from original logits at the answer index
            batch_label = original_sliced.gather(
                dim=-1, index=answers_idxs.unsqueeze(-1)
            ).squeeze(-1)
        else:
            batch_label = original_logits.squeeze()[pos_idx.as_index]
        if labels is not None:
            labels = torch.cat([labels, batch_label], dim=0)
        else:
            labels = batch_label

    logit_lens_results.keys()
    return logit_lens_results, labels


# @dataclass
# class TunedLensConfig:
#     """
#     Configuration for trainin tuned lens maps

#     Args:
#         num_epochs: number of epochs to train
#         lr: learning rate
#         from_activation: whether to train from activations. If False, train from resid stacks (default: True)
#         to_logits: whether to train to logits. If False, train to hook_resid_post of final layer (default: True)
#         pos_slice: slice to apply to the positional dimension. Default is to exclude the BOS token (slice(1, None, None))
#     """

#     num_epochs: int
#     lr: float
#     from_activation: bool = True
#     to_logits: bool = True
#     pos_slice: slice = slice(1, None, None)


# def do_tuned_lens(
#     model_pair: mp.BaseModelPair,
#     loader: torch.utils.data.DataLoader,
#     config: TunedLensConfig = TunedLensConfig(3, 1e-3),
# ):
#     # make translator linear maps for each node
#     pass