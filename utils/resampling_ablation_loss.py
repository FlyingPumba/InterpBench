from functools import partial
from typing import List

import torch as t
from jaxtyping import Float
from torch import Tensor
from transformer_lens import HookedTransformer, ActivationCache
from transformer_lens.hook_points import HookPoint

from utils.hooked_tracr_transformer import HookedTracrTransformerBatchInput


def get_resampling_ablation_loss(
    clean_inputs: HookedTracrTransformerBatchInput,
    corrupted_inputs: HookedTracrTransformerBatchInput,
    base_model: HookedTransformer,
    hypothesis_model: HookedTransformer,
    intervention_filters: List[str] = ["hook_embed", "hook_attn_out", "hook_mlp_out"]
) -> Float[Tensor, ""]:
  # we assume that both models have the same architecture. Otherwise, the comparison is flawed since they have diffenret
  # intervention points.
  assert base_model.cfg.n_layers == hypothesis_model.cfg.n_layers
  assert base_model.cfg.n_heads == hypothesis_model.cfg.n_heads
  assert base_model.cfg.n_ctx == hypothesis_model.cfg.n_ctx
  assert base_model.cfg.d_vocab == hypothesis_model.cfg.d_vocab

  # first, we run the corrupted inputs on both models and save the activation caches.
  _, base_model_corrupted_cache = base_model.run_with_cache(corrupted_inputs)
  _, hypothesis_model_corrupted_cache = hypothesis_model.run_with_cache(corrupted_inputs)

  # for each intervention point in both models
  base_model_outputs = []
  hypothesis_model_outputs = []
  for hook_name, hook in base_model.hook_dict.items():
    assert hook_name in hypothesis_model.hook_dict, f"hook {hook_name} not found in hypothesis model."

    if intervention_filters is not None and not any([filter in hook_name for filter in intervention_filters]):
      # skip this hook point
      continue

    def corrupted_output_hook_fn(
        residual_stream: Float[Tensor, "batch seq_len d_model"],
        hook: HookPoint,
        corrupted_cache: ActivationCache = None
    ):
      # We will replace the output residual_stream with an average of the corrupted ones.
      # However, we may have different batch sizes, so we need to take care of that.
      avg_corrupted_residual_stream = t.mean(corrupted_cache[hook.name], dim=0)

      # We repeat the average corrupted residual stream to match the batch size of the clean inputs.
      new_residual_stream = avg_corrupted_residual_stream.repeat(residual_stream.shape[0], 1, 1)
      return new_residual_stream

    # We intervene both models at the same point, run them on the clean data and save the output.
    with base_model.hooks(fwd_hooks=[(hook_name,
                                      partial(corrupted_output_hook_fn,
                                              corrupted_cache=base_model_corrupted_cache))]):
      with hypothesis_model.hooks(fwd_hooks=[(hook_name,
                                              partial(corrupted_output_hook_fn,
                                                      corrupted_cache=hypothesis_model_corrupted_cache))]):
        base_model_logits = base_model(clean_inputs)
        hypothesis_model_logits = hypothesis_model(clean_inputs)

        base_model_outputs.append(base_model_logits)
        hypothesis_model_outputs.append(hypothesis_model_logits)

  # compare the outputs of both models, e.g., using KL divergence
  base_model_outputs = t.cat(base_model_outputs, dim=0)
  hypothesis_model_outputs = t.cat(hypothesis_model_outputs, dim=0)

  # use MSE to compare the outputs of both models
  loss = t.nn.functional.mse_loss(base_model_outputs, hypothesis_model_outputs)

  return loss
