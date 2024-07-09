from jaxtyping import Float
from transformer_lens import ActivationCache
from fancy_einsum import einsum
import iit.utils.index as index
import torch
from iit.model_pairs.base_model_pair import BaseModelPair


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


def do_logit_lens(model_pair: BaseModelPair, loader: torch.utils.data.DataLoader):
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
            # control_logit = model.unembed.W_U[:, 0]
            answers_idxs = original_logits.argmax(dim=-1)[pos_idx.as_index]
            logit_diff_directions = model.unembed.W_U.T[answers_idxs] #- control_logit
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
            # control_logit = original_sliced[:, :, 0]
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

    return logit_lens_results, labels


def do_logit_lens_per_vocab_idx(model_pair: BaseModelPair, loader: torch.utils.data.DataLoader):
    model = model_pair.ll_model
    logit_lens_results = {}
    labels = {}
    for i in range(model.cfg.d_vocab_out):
        labels[i] = None

    assert model_pair.hl_model.is_categorical()
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
        per_layer_scaled_residual_stack = cache.apply_ln_to_stack(
            per_layer_residual, layer=-1, pos_slice=pos_slice
        )
        per_head_scaled_residual_stack = cache.apply_ln_to_stack(
            per_head_residual, layer=-1, pos_slice=pos_slice
        )
        logit_diff_directions = model.unembed.W_U.T # d_model, d_vocab_out
        # make batch, pos, d_model, d_vocab_out by expanding
        batch_dims = per_layer_scaled_residual_stack.shape[1]
        pos_dims = per_layer_scaled_residual_stack.shape[2]
        logit_diff_directions = logit_diff_directions.unsqueeze(0).unsqueeze(1).expand(
            batch_dims, pos_dims, -1, -1
        )
        # logit lens
        per_layer_logit_diffs = einsum(
            "... batch pos d_model, batch pos d_vocab_out d_model -> ... batch pos d_vocab_out",
            per_layer_scaled_residual_stack,
            logit_diff_directions
        )
        per_head_logit_diffs = einsum(
            "... batch pos d_model, batch pos d_vocab_out d_model -> ... batch pos d_vocab_out",
            per_head_scaled_residual_stack,
            logit_diff_directions
        )
        for i in range(model.cfg.d_vocab_out):
            per_layer_logit_diff = per_layer_logit_diffs[..., i]
            per_head_logit_diff = per_head_logit_diffs[..., i]
            
            label = original_logits[pos_idx.as_index][:, :, i]
            # store results
            for layer, logit_diff in zip(layers, per_layer_logit_diff):
                if layer not in logit_lens_results:
                    logit_lens_results[layer] = {}
                if i not in logit_lens_results[layer]:
                    logit_lens_results[layer][i] = logit_diff
                else:
                    # stack the results at dim 0
                    logit_lens_results[layer][i] = torch.cat(
                        [logit_lens_results[layer][i], logit_diff], dim=0
                    )
            
            for attn, logit_diff in zip(attns, per_head_logit_diff):
                if attn not in logit_lens_results:
                    logit_lens_results[attn] = {}
                if i not in logit_lens_results[attn]:
                    logit_lens_results[attn][i] = logit_diff
                else:
                    logit_lens_results[attn][i] = torch.cat(
                        [logit_lens_results[attn][i], logit_diff], dim=0
                    )
            
            if labels[i] is not None:
                labels[i] = torch.cat([labels[i], label], dim=0)
            else:
                labels[i] = label

    return logit_lens_results, labels
