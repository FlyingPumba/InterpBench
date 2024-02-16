from contextlib import contextmanager
from functools import partial
from typing import List

from jaxtyping import Float
from torch import Tensor
from transformer_lens import ActivationCache, HookedTransformer
from transformer_lens.hook_points import HookPoint

from training.compression.autencoder import AutoEncoder
from utils.hooked_tracr_transformer import HookedTracrTransformerBatchInput
from utils.resampling_ablation_loss.intervention_type import InterventionType


def regular_intervention_hook_fn(
    residual_stream: Float[Tensor, "batch seq_len d_model"],
    hook: HookPoint,
    corrupted_cache: ActivationCache = None
):
  """This hook just replaces the output with a corrupted output."""
  return corrupted_cache[hook.name]


def encoder_intervention_hook_fn(
    residual_stream: Float[Tensor, "batch seq_len d_model"],
    hook: HookPoint,
    corrupted_cache: ActivationCache = None,
    autoencoder: AutoEncoder = None
):
  """This hook replaces the output with a corrupted output passed through the encoder."""
  return autoencoder.encoder(corrupted_cache[hook.name])


def decoder_intervention_hook_fn(
    residual_stream: Float[Tensor, "batch seq_len d_model"],
    hook: HookPoint,
    corrupted_cache: ActivationCache = None,
    autoencoder: AutoEncoder = None
):
  """This hook replaces the output with a corrupted output passed through the decoder."""
  return autoencoder.decoder(corrupted_cache[hook.name])


class Intervention(object):
  def __init__(self,
               hook_names: List[str],
               hook_intervention_types: List[InterventionType],
               autoencoder: AutoEncoder | None = None):
    self.hook_names = hook_names
    self.hook_intervention_types = hook_intervention_types
    self.autoencoder = autoencoder

    assert len(hook_names) == len(hook_intervention_types), \
      "hook_names and hook_intervention_types should have the same length."

    # Assert there are no interventions that require an autoencoder if it is not provided.
    if autoencoder is None:
      for intervention_type in hook_intervention_types:
        assert intervention_type not in [InterventionType.CORRUPTED_ENCODING,
                                         InterventionType.CORRUPTED_DECODING,
                                         InterventionType.CLEAN_ENCODING,
                                         InterventionType.CLEAN_DECODING], \
          "Autoencoder is not provided, so interventions that require it are not allowed."

    self.has_clean_interventions = any([intervention_type in [InterventionType.CLEAN_ENCODING,
                                                              InterventionType.CLEAN_DECODING]
                                        for intervention_type in hook_intervention_types])

  @contextmanager
  def hooks(self,
            base_model: HookedTransformer,
            hypothesis_model: HookedTransformer,
            clean_inputs_batch: HookedTracrTransformerBatchInput,
            corrupted_inputs_batch: HookedTracrTransformerBatchInput):

    # Run the corrupted inputs on both models and save the activation caches.
    _, base_model_corrupted_cache = base_model.run_with_cache(corrupted_inputs_batch)
    _, hypothesis_model_corrupted_cache = hypothesis_model.run_with_cache(corrupted_inputs_batch)

    base_model_clean_cache = None
    hypothesis_model_clean_cache = None
    if self.has_clean_interventions:
      # Run the clean inputs on both models and save the activation caches.
      _, base_model_clean_cache = base_model.run_with_cache(clean_inputs_batch)
      _, hypothesis_model_clean_cache = hypothesis_model.run_with_cache(clean_inputs_batch)

    base_model_hooks = []
    hypothesis_model_hooks = []

    for hook_name, intervention_type in zip(self.hook_names, self.hook_intervention_types):
      if intervention_type == InterventionType.REGULAR_CORRUPTED:
        base_model_hooks.append((hook_name, partial(regular_intervention_hook_fn,
                                                    corrupted_cache=base_model_corrupted_cache)))
        hypothesis_model_hooks.append((hook_name, partial(regular_intervention_hook_fn,
                                                          corrupted_cache=hypothesis_model_corrupted_cache)))

      elif intervention_type == InterventionType.CORRUPTED_ENCODING:
        base_model_hooks.append((hook_name, partial(regular_intervention_hook_fn,
                                                    corrupted_cache=base_model_corrupted_cache)))
        hypothesis_model_hooks.append((hook_name, partial(encoder_intervention_hook_fn,
                                                          corrupted_cache=base_model_corrupted_cache,
                                                          autoencoder=self.autoencoder)))

      elif intervention_type == InterventionType.CORRUPTED_DECODING:
        base_model_hooks.append((hook_name, partial(decoder_intervention_hook_fn,
                                                    corrupted_cache=hypothesis_model_corrupted_cache,
                                                    autoencoder=self.autoencoder)))
        hypothesis_model_hooks.append((hook_name, partial(regular_intervention_hook_fn,
                                                          corrupted_cache=hypothesis_model_corrupted_cache)))

      elif intervention_type == InterventionType.CLEAN_ENCODING:
        hypothesis_model_hooks.append((hook_name, partial(encoder_intervention_hook_fn,
                                                          corrupted_cache=base_model_clean_cache,
                                                          autoencoder=self.autoencoder)))

      elif intervention_type == InterventionType.CLEAN_DECODING:
        base_model_hooks.append((hook_name, partial(decoder_intervention_hook_fn,
                                                    corrupted_cache=hypothesis_model_clean_cache,
                                                    autoencoder=self.autoencoder)))

      elif intervention_type == InterventionType.NO_INTERVENTION:
        # No hooks to add
        pass

      else:
        raise ValueError(f"Intervention type {intervention_type} is not supported.")

    with base_model.hooks(base_model_hooks):
      with hypothesis_model.hooks(hypothesis_model_hooks):
        yield self
