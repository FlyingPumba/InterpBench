import gc
from dataclasses import dataclass
from typing import List, Dict

import numpy as np
import torch as t
from jaxtyping import Float
from torch import Tensor
from transformer_lens import HookedTransformer

from circuits_benchmark.benchmark.case_dataset import CaseDataset
from circuits_benchmark.metrics.resampling_ablation_loss.intervention import InterventionData
from circuits_benchmark.metrics.resampling_ablation_loss.resample_ablation_interventions import get_interventions
from circuits_benchmark.training.compression.activation_mapper.activation_mapper import ActivationMapper
from circuits_benchmark.training.compression.activation_mapper.multi_hook_activation_mapper import MultiHookActivationMapper


@dataclass
class ResampleAblationLossOutput:
  loss: Float[Tensor, ""]
  variance_explained: Float[Tensor, ""]
  max_loss_per_hook: Dict[str, Float[Tensor, ""]]
  mean_loss_per_hook: Dict[str, Float[Tensor, ""]]
  intervened_hooks: List[str]


def get_resample_ablation_loss_from_inputs(
    clean_inputs: CaseDataset,
    corrupted_inputs: CaseDataset,
    base_model: HookedTransformer,
    hypothesis_model: HookedTransformer,
    activation_mapper: MultiHookActivationMapper | ActivationMapper | None = None,
    hook_filters: List[str] | None = None,
    batch_size: int = 2048,
    max_interventions: int = 10,
    max_components: int = 1,
    is_categorical: bool = False,
) -> ResampleAblationLossOutput:
  # assert that clean_input and corrupted_input have the same length
  assert len(clean_inputs) == len(corrupted_inputs), "clean and corrupted inputs should have same length."
  # assert that clean and corrupted inputs are not exactly the same, otherwise the comparison is flawed.
  assert clean_inputs != corrupted_inputs, "clean and corrupted inputs should have different data."

  # Build data for interventions before starting to avoid recomputing the same data for each intervention.
  batched_intervention_data = get_batched_intervention_data(clean_inputs,
                                                            corrupted_inputs,
                                                            base_model,
                                                            hypothesis_model,
                                                            activation_mapper,
                                                            batch_size)

  return get_resample_ablation_loss(batched_intervention_data, base_model, hypothesis_model, activation_mapper,
                                    hook_filters, max_interventions, max_components, is_categorical)


def get_resample_ablation_loss(batched_intervention_data: List[InterventionData],
                               base_model: HookedTransformer,
                               hypothesis_model: HookedTransformer,
                               activation_mapper: MultiHookActivationMapper | ActivationMapper | None,
                               hook_filters: List[str] | None = None,
                               max_interventions: int = 10,
                               max_components: int = 1,
                               is_categorical: bool = False) -> ResampleAblationLossOutput:
  # This is a memory intensive operation, so we will garbage collect before starting.
  gc.collect()
  t.cuda.empty_cache()

  if hook_filters is None:
    if activation_mapper is None or isinstance(activation_mapper, ActivationMapper):
      # by default, we use the following hooks for the intervention points.
      # This will give 2 + n_layers * 2 intervention points.
      hook_filters = ["hook_embed", "hook_pos_embed", "hook_attn_out", "hook_mlp_out"]
    else:
      # We use all hook names that can be processed by the multi activation mapper.
      hook_filters = [k for k in base_model.hook_dict.keys() if activation_mapper.supports_hook(k)]

  # we assume that both models have the same architecture. Otherwise, the comparison is flawed since they have different
  # intervention points.
  assert base_model.cfg.n_layers == hypothesis_model.cfg.n_layers
  assert base_model.cfg.n_heads == hypothesis_model.cfg.n_heads
  assert base_model.cfg.n_ctx == hypothesis_model.cfg.n_ctx
  assert base_model.cfg.d_vocab == hypothesis_model.cfg.d_vocab

  assert max_interventions > 0, "max_interventions should be greater than 0."

  # Calculate the variance of the base model logits.
  base_model_logits_variance = []
  for intervention_data in batched_intervention_data:
    clean_inputs_batch = intervention_data.clean_inputs
    base_model_logits = base_model(clean_inputs_batch)
    base_model_logits_variance.append(t.var(base_model_logits).item())
  base_model_logits_variance = np.mean(base_model_logits_variance)

  # for each intervention, run both models, calculate MSE and add it to the losses.
  losses = []
  variance_explained = []
  max_loss_per_hook = {}
  mean_loss_per_hook = {}
  intervened_hooks = set()
  for intervention in get_interventions(base_model,
                                        hypothesis_model,
                                        hook_filters,
                                        activation_mapper,
                                        max_interventions,
                                        max_components):
    # We may have more than one batch of inputs, so we need to iterate over them, and average at the end.
    batched_data_intervention_losses = []
    batched_data_intervention_variance_explained = []
    for intervention_data in batched_intervention_data:
      clean_inputs_batch = intervention_data.clean_inputs

      with intervention.hooks(base_model, hypothesis_model, intervention_data):
        if is_categorical:
          # use cross entropy loss for categorical outputs.
          base_model_logits = base_model(clean_inputs_batch)
          hypothesis_model_logits = hypothesis_model(clean_inputs_batch)

          # The output has unspecified behavior for the BOS token, so we discard it on the loss calculation.
          base_model_logits = base_model_logits[:, 1:]
          hypothesis_model_logits = hypothesis_model_logits[:, 1:]

          log_probs = hypothesis_model_logits.log_softmax(dim=-1)
          expected_tokens = base_model_logits.argmax(dim=-1)

          # Get logprobs the first seq_len-1 predictions (so we can compare them with the actual next tokens)
          log_probs_for_tokens = log_probs.gather(dim=-1, index=expected_tokens.unsqueeze(-1)).squeeze(-1)

          loss = -log_probs_for_tokens.mean()
        else:
          # Use MSE loss for numerical outputs.
          base_model_logits = base_model(clean_inputs_batch)
          hypothesis_model_logits = hypothesis_model(clean_inputs_batch)
          loss = t.nn.functional.mse_loss(base_model_logits, hypothesis_model_logits)

        var_explained = 1 - loss / base_model_logits_variance

        batched_data_intervention_losses.append(loss.reshape(1))
        batched_data_intervention_variance_explained.append(var_explained.reshape(1))

    intervention_loss = t.cat(batched_data_intervention_losses).mean().reshape(1)
    losses.append(intervention_loss)

    intervention_var_exp = t.cat(batched_data_intervention_variance_explained).mean().reshape(1)
    variance_explained.append(intervention_var_exp)

    intervened_hooks.update(intervention.get_intervened_hooks())

    # store the max and mean loss per hook for the intervention.
    for hook_name in intervention.get_intervened_hooks():
      if hook_name in max_loss_per_hook:
        max_loss_per_hook[hook_name] = max(max_loss_per_hook[hook_name], intervention_loss)
      else:
        max_loss_per_hook[hook_name] = intervention_loss

      if hook_name in mean_loss_per_hook:
        mean_loss_per_hook[hook_name] = t.cat([mean_loss_per_hook[hook_name], intervention_loss])
      else:
        mean_loss_per_hook[hook_name] = intervention_loss

  for hook_name in mean_loss_per_hook:
    mean_loss_per_hook[hook_name] = mean_loss_per_hook[hook_name].mean()

  return ResampleAblationLossOutput(
    loss=t.cat(losses).mean(),
    variance_explained=t.cat(variance_explained).mean(),
    max_loss_per_hook=max_loss_per_hook,
    mean_loss_per_hook=mean_loss_per_hook,
    intervened_hooks=list(intervened_hooks)
  )


def get_batched_intervention_data(
    clean_inputs: CaseDataset,
    corrupted_inputs: CaseDataset,
    base_model: HookedTransformer,
    hypothesis_model: HookedTransformer,
    activation_mapper: MultiHookActivationMapper | ActivationMapper | None = None,
    batch_size: int = 2048,
) -> List[InterventionData]:
  data = []

  for clean_inputs_batch, corrupted_inputs_batch in zip(clean_inputs.get_inputs_loader(batch_size),
                                                        corrupted_inputs.get_inputs_loader(batch_size)):
    clean_inputs_batch = clean_inputs_batch[CaseDataset.INPUT_FIELD]
    corrupted_inputs_batch = corrupted_inputs_batch[CaseDataset.INPUT_FIELD]

    # Run the corrupted inputs on both models and save the activation caches.
    _, base_model_corrupted_cache = base_model.run_with_cache(corrupted_inputs_batch)
    _, hypothesis_model_corrupted_cache = hypothesis_model.run_with_cache(corrupted_inputs_batch)

    base_model_clean_cache = None
    hypothesis_model_clean_cache = None
    if activation_mapper is not None:
      # Run the clean inputs on both models and save the activation caches.
      _, base_model_clean_cache = base_model.run_with_cache(clean_inputs_batch)
      _, hypothesis_model_clean_cache = hypothesis_model.run_with_cache(clean_inputs_batch)

    intervention_data = InterventionData(clean_inputs_batch,
                                         base_model_corrupted_cache,
                                         hypothesis_model_corrupted_cache,
                                         base_model_clean_cache,
                                         hypothesis_model_clean_cache)
    data.append(intervention_data)

  return data
