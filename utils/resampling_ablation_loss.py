import random
from functools import partial
from typing import List

import numpy as np
import torch as t
from jaxtyping import Float
from torch import Tensor
from transformer_lens import HookedTransformer, ActivationCache
from transformer_lens.hook_points import HookPoint

from benchmark.case_dataset import CaseDataset
from training.compression.autencoder import AutoEncoder


def get_resampling_ablation_loss(
    clean_inputs: CaseDataset,
    corrupted_inputs: CaseDataset,
    base_model: HookedTransformer,
    hypothesis_model: HookedTransformer,
    autoencoder: AutoEncoder | None = None,
    hook_filters: List[str] = ["hook_embed", "hook_pos_embed", "hook_attn_out", "hook_mlp_out"],
    batch_size: int = 2048,
    intervention_samples_ratio: float|None = 0.3
) -> Float[Tensor, ""]:
  # we assume that both models have the same architecture. Otherwise, the comparison is flawed since they have different
  # intervention points.
  assert base_model.cfg.n_layers == hypothesis_model.cfg.n_layers
  assert base_model.cfg.n_heads == hypothesis_model.cfg.n_heads
  assert base_model.cfg.n_ctx == hypothesis_model.cfg.n_ctx
  assert base_model.cfg.d_vocab == hypothesis_model.cfg.d_vocab

  # assert that clean_input and corrupted_input have the same length
  assert len(clean_inputs) == len(corrupted_inputs), "clean and corrupted inputs should have same length."

  # assert that clean and corrupted inputs are not exactly the same, otherwise the comparison is flawed.
  assert clean_inputs != corrupted_inputs, "clean and corrupted inputs should have different data."

  combinations = build_intervention_points(base_model, hook_filters, autoencoder)
  assert len(combinations) > 0, "No valid intervention points found."

  if intervention_samples_ratio is not None:
    # if we have a limit on the number of intervention samples, we sample a subset of the combinations (no replacement).
    assert 0 < intervention_samples_ratio <= 1, "intervention_samples_ratio should be between 0 and 1."
    intervention_samples = max(int(intervention_samples_ratio * len(combinations)), 1)
    combinations = random.sample(combinations, intervention_samples)

  losses = []
  for clean_inputs_batch, corrupted_inputs_batch in zip(clean_inputs.get_inputs_loader(batch_size),
                                                        corrupted_inputs.get_inputs_loader(batch_size)):
    clean_inputs_batch = clean_inputs_batch[CaseDataset.INPUT_FIELD]
    corrupted_inputs_batch = corrupted_inputs_batch[CaseDataset.INPUT_FIELD]

    # first, we run the corrupted inputs on both models and save the activation caches.
    _, base_model_corrupted_cache = base_model.run_with_cache(corrupted_inputs_batch)
    _, hypothesis_model_corrupted_cache = hypothesis_model.run_with_cache(corrupted_inputs_batch)

    # for each intervention combination, run both models, calculate MSE and add it to the losses
    for (regular_base_hook_name, regular_hypothesis_hook_name, encoder_hook_name, decoder_hook_name) in combinations:
      assert regular_base_hook_name in base_model.hook_dict, \
        f"hook {regular_base_hook_name} not found in base model."
      assert regular_hypothesis_hook_name in hypothesis_model.hook_dict, \
        f"hook {regular_hypothesis_hook_name} not found in hypothesis model."
      assert (encoder_hook_name is None or encoder_hook_name in hypothesis_model.hook_dict), \
        f"hook {encoder_hook_name} not found in hypothesis model."
      assert (decoder_hook_name is None or decoder_hook_name in base_model.hook_dict), \
        f"hook {decoder_hook_name} not found in base model."

      # We intervene the models according to the provided hook names, run them, and calculate the MSE.
      base_model_hooks = []
      hypothesis_model_hooks = []

      if regular_base_hook_name is not None:
        base_model_hooks.append((regular_base_hook_name,
                                 partial(regular_corrupted_intervention_hook_fn,
                                         corrupted_cache=base_model_corrupted_cache)))

      if regular_hypothesis_hook_name is not None:
        hypothesis_model_hooks.append((regular_hypothesis_hook_name,
                                       partial(regular_corrupted_intervention_hook_fn,
                                               corrupted_cache=hypothesis_model_corrupted_cache)))

      if encoder_hook_name is not None:
        hypothesis_model_hooks.append((encoder_hook_name, partial(encoder_corrupted_intervention_hook_fn,
                                                                  corrupted_cache=base_model_corrupted_cache,
                                                                  autoencoder=autoencoder)))
      if decoder_hook_name is not None:
        base_model_hooks.append((decoder_hook_name, partial(decoder_corrupted_intervention_hook_fn,
                                                            corrupted_cache=hypothesis_model_corrupted_cache,
                                                            autoencoder=autoencoder)))

      with base_model.hooks(fwd_hooks=base_model_hooks):
        with hypothesis_model.hooks(fwd_hooks=hypothesis_model_hooks):
          base_model_logits = base_model(clean_inputs_batch)
          hypothesis_model_logits = hypothesis_model(clean_inputs_batch)

          loss = t.nn.functional.mse_loss(base_model_logits, hypothesis_model_logits).item()
          losses.append(loss)

  return np.mean(losses)


def build_intervention_points(base_model, hook_filters, autoencoder):
  """Builds the different combinations of intervention points for the base model and the hypothesis model.
  We have 4 different types of interventions:
  - A. Regular patching on the base model: we replace the output of the hook with the corrupted output.
  - B. Regular patching on the hypothesis model: we replace the output of the hook with the corrupted output.
  - C. Encoder patching on the hypothesis model: we replace the output of the hook with the corrupted output from base
    model passed through the encoder.
  - D. Decoder patching on the base model: we replace the output of the hook with the corrupted output from hypothesis
    model passed through the decoder.

  Important: we also want to avoid conflicting interventions, e.g., patching the same hook twice by combining A and D
    interventions, or B and C interventions.
  """
  hook_names: List[str | None] = list(base_model.hook_dict.keys())
  hook_names_for_regular_patching = hook_names + [None]

  if autoencoder is not None:
    hook_names_for_encoder_decoder_patching = hook_names_for_regular_patching
  else:
    hook_names_for_encoder_decoder_patching = [None]

  combinations = []
  for regular_base_hook_name in hook_names:
    for regular_hypothesis_hook_name in hook_names:
      for encoder_hook_name in hook_names_for_encoder_decoder_patching:
        for decoder_hook_name in hook_names_for_encoder_decoder_patching:
          hook_name_candidates = [regular_base_hook_name, regular_hypothesis_hook_name, encoder_hook_name,
                                  decoder_hook_name]
          if any(should_hook_name_be_skipped_due_to_filters(hook_name, hook_filters)
                 for hook_name in hook_name_candidates):
            continue

          # check for conflicts
          if regular_base_hook_name is not None and decoder_hook_name is not None and \
              regular_base_hook_name == decoder_hook_name:
            continue

          if regular_hypothesis_hook_name is not None and encoder_hook_name is not None and \
              regular_hypothesis_hook_name == encoder_hook_name:
            continue

          combinations.append((regular_base_hook_name,
                               regular_hypothesis_hook_name,
                               encoder_hook_name,
                               decoder_hook_name))

  return combinations


def should_hook_name_be_skipped_due_to_filters(hook_name: str | None, hook_filters: List[str]) -> bool:
  if hook_filters is None:
    # No filters to apply
    return False

  if hook_name is None:
    # No hook name to apply the filters to
    return False

  return not any([filter in hook_name for filter in hook_filters])


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
    corrupted_cache: ActivationCache = None,
    autoencoder: AutoEncoder = None
):
  """This hook replaces the output with a corrupted output passed through the encoder."""
  return autoencoder.encoder(corrupted_cache[hook.name])


def decoder_corrupted_intervention_hook_fn(
    residual_stream: Float[Tensor, "batch seq_len d_model"],
    hook: HookPoint,
    corrupted_cache: ActivationCache = None,
    autoencoder: AutoEncoder = None
):
  """This hook replaces the output with a corrupted output passed through the decoder."""
  return autoencoder.decoder(corrupted_cache[hook.name])
