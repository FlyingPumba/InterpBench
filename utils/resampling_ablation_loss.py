from functools import partial
from typing import List

import torch as t
from jaxtyping import Float
from torch import Tensor
from transformer_lens import HookedTransformer, ActivationCache
from transformer_lens.hook_points import HookPoint

from training.compression.autencoder import AutoEncoder
from utils.hooked_tracr_transformer import HookedTracrTransformerBatchInput


def get_resampling_ablation_loss(
    clean_inputs: HookedTracrTransformerBatchInput,
    corrupted_inputs: HookedTracrTransformerBatchInput,
    base_model: HookedTransformer,
    hypothesis_model: HookedTransformer,
    autoencoder: AutoEncoder | None = None,
    hook_filters: List[str] = ["hook_embed", "hook_pos_embed", "hook_attn_out", "hook_mlp_out"],
) -> Float[Tensor, ""]:
  # we assume that both models have the same architecture. Otherwise, the comparison is flawed since they have different
  # intervention points.
  assert base_model.cfg.n_layers == hypothesis_model.cfg.n_layers
  assert base_model.cfg.n_heads == hypothesis_model.cfg.n_heads
  assert base_model.cfg.n_ctx == hypothesis_model.cfg.n_ctx
  assert base_model.cfg.d_vocab == hypothesis_model.cfg.d_vocab

  # assert that clean and corrupted inputs are not exactly the same, otherwise the comparison is flawed.
  all_equal = all(clean_input == corrupted_input
                  for clean_input, corrupted_input in zip(clean_inputs, corrupted_inputs))
  assert not all_equal, "clean and corrupted inputs are exactly the same. This is not a valid comparison."

  # assert that clean_input and corrupted_input have the same length
  assert len(clean_inputs) == len(corrupted_inputs), "clean and corrupted inputs have different lengths."

  # first, we run the corrupted inputs on both models and save the activation caches.
  _, base_model_corrupted_cache = base_model.run_with_cache(corrupted_inputs)
  _, hypothesis_model_corrupted_cache = hypothesis_model.run_with_cache(corrupted_inputs)

  # Define an auxiliary function to check if a hook name should be skipped based on the hook filters
  def should_hook_name_be_skipped_due_to_filters(hook_name: str) -> bool:
    if hook_filters is None:
      # No filters to apply
      return False

    if hook_name is None:
      # No hook name to apply the filters to
      return False

    return not any([filter in hook_name for filter in hook_filters])

  # we have P different hooks to work on, each time we choose one for the regular patching, we are left with P-1 hooks
  # for the encoder-patching and P-1 hooks for the decoder-patching. If we also add the possibility of not using
  # encoder-patching or decoder-patching, we have P^3 different combinations.
  hook_names: List[str|None] = list(base_model.hook_dict.keys())

  if autoencoder is not None:
    hook_names_for_encoder_decoder_patching = hook_names + [None]
  else:
    hook_names_for_encoder_decoder_patching = [None]

  combinations = []
  for regular_hook_name in hook_names:
    for encoder_hook_name in hook_names_for_encoder_decoder_patching:
      for decoder_hook_name in hook_names_for_encoder_decoder_patching:
        if should_hook_name_be_skipped_due_to_filters(regular_hook_name):
          continue
        if should_hook_name_be_skipped_due_to_filters(encoder_hook_name):
          continue
        if should_hook_name_be_skipped_due_to_filters(decoder_hook_name):
          continue

        if encoder_hook_name is not None and regular_hook_name == encoder_hook_name:
          # we don't want to patch the same hook twice
          continue
        if decoder_hook_name is not None and regular_hook_name == decoder_hook_name:
          # we don't want to patch the same hook twice
          continue

        combinations.append((regular_hook_name, encoder_hook_name, decoder_hook_name))

  # for each intervention combination, run both models and collect the outputs
  base_model_outputs = []
  hypothesis_model_outputs = []
  for (regular_hook_name, encoder_hook_name, decoder_hook_name) in combinations:
    assert regular_hook_name in hypothesis_model.hook_dict, f"hook {regular_hook_name} not found in hypothesis model."
    assert (encoder_hook_name is None or
            encoder_hook_name in hypothesis_model.hook_dict), f"hook {encoder_hook_name} not found in hypothesis model."
    assert (decoder_hook_name is None or
            decoder_hook_name in hypothesis_model.hook_dict), f"hook {decoder_hook_name} not found in hypothesis model."

    def regular_corrupted_intervention_hook_fn(
        residual_stream: Float[Tensor, "batch seq_len d_model"],
        hook: HookPoint,
        corrupted_cache: ActivationCache = None
    ):
      """This hook just replaces the output with a corrupted output."""
      return corrupted_cache[hook.name]

    def encoder_corrupted_intervention_hook_fn(
        residual_stream: Float[Tensor, "batch seq_len d_model"],
        hook: HookPoint,
        corrupted_cache: ActivationCache = None
    ):
      """This hook replaces the output with a corrupted output passed through the encoder."""
      return autoencoder.encoder(corrupted_cache[hook.name])

    def decoder_corrupted_intervention_hook_fn(
        residual_stream: Float[Tensor, "batch seq_len d_model"],
        hook: HookPoint,
        corrupted_cache: ActivationCache = None
    ):
      """This hook replaces the output with a corrupted output passed through the decoder."""
      return autoencoder.decoder(corrupted_cache[hook.name])

    # We intervene both models at the same point, run them on the clean data and save the output.
    base_model_hooks = [(regular_hook_name, partial(regular_corrupted_intervention_hook_fn,
                                                    corrupted_cache=base_model_corrupted_cache))]
    hypothesis_model_hooks = [(regular_hook_name, partial(regular_corrupted_intervention_hook_fn,
                                                          corrupted_cache=hypothesis_model_corrupted_cache))]

    if encoder_hook_name is not None:
      hypothesis_model_hooks.append((encoder_hook_name, partial(encoder_corrupted_intervention_hook_fn,
                                                                corrupted_cache=base_model_corrupted_cache)))
    if decoder_hook_name is not None:
      base_model_hooks.append((decoder_hook_name, partial(decoder_corrupted_intervention_hook_fn,
                                                          corrupted_cache=hypothesis_model_corrupted_cache)))

    with base_model.hooks(fwd_hooks=base_model_hooks):
      with hypothesis_model.hooks(fwd_hooks=hypothesis_model_hooks):
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
