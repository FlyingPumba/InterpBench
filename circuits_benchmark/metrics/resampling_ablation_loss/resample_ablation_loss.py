import random
from typing import List, Generator

import numpy as np
import torch as t
from jaxtyping import Float
from torch import Tensor
from transformer_lens import HookedTransformer

from circuits_benchmark.benchmark.case_dataset import CaseDataset
from circuits_benchmark.metrics.resampling_ablation_loss.intervention import Intervention, InterventionData
from circuits_benchmark.metrics.resampling_ablation_loss.intervention_type import InterventionType
from circuits_benchmark.training.compression.residual_stream_mapper.residual_stream_mapper import ResidualStreamMapper


def get_resample_ablation_loss(
    clean_inputs: CaseDataset,
    corrupted_inputs: CaseDataset,
    base_model: HookedTransformer,
    hypothesis_model: HookedTransformer,
    residual_stream_mapper: ResidualStreamMapper | None = None,
    hook_filters: List[str] | None = None,
    batch_size: int = 2048,
    max_interventions: int = 10
) -> Float[Tensor, ""]:
  if hook_filters is None:
    # by default, we use the following hooks for the intervention points.
    # This will give 2 + n_layers * 2 intervention points.
    hook_filters = ["hook_embed", "hook_pos_embed", "hook_attn_out", "hook_mlp_out"]

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
  assert max_interventions > 0, "max_interventions should be greater than 0."

  # Build data for interventions before starting to avoid recomputing the same data for each intervention.
  batched_intervention_data = get_batched_intervention_data(clean_inputs,
                                                            corrupted_inputs,
                                                            base_model,
                                                            hypothesis_model,
                                                            residual_stream_mapper,
                                                            batch_size)

  # for each intervention, run both models, calculate MSE and add it to the losses.
  losses = []
  for intervention in get_interventions(base_model,
                                        hypothesis_model,
                                        hook_filters,
                                        residual_stream_mapper,
                                        max_interventions):
    # We may have more than one batch of inputs, so we need to iterate over them, and average at the end.
    intervention_losses = []
    for intervention_data in batched_intervention_data:
      clean_inputs_batch = intervention_data.clean_inputs

      with intervention.hooks(base_model, hypothesis_model, intervention_data):
        base_model_logits = base_model(clean_inputs_batch)
        hypothesis_model_logits = hypothesis_model(clean_inputs_batch)

        loss = t.nn.functional.mse_loss(base_model_logits, hypothesis_model_logits).item()
        intervention_losses.append(loss)

    losses.append(np.mean(intervention_losses))

  return np.mean(losses)


def get_interventions(
    base_model: HookedTransformer,
    hypothesis_model: HookedTransformer,
    hook_filters: List[str],
    residual_stream_mapper: ResidualStreamMapper | None = None,
    max_interventions: int = 10) -> Generator[Intervention, None, None]:
  """Builds the different combinations for possible interventions on the base and hypothesis models."""
  hook_names: List[str | None] = list(base_model.hook_dict.keys())
  hook_names_for_patching = [name for name in hook_names
                             if not should_hook_name_be_skipped_due_to_filters(name, hook_filters)]

  # assert all hook names for patching are also present in the hypothesis model
  assert all([hook_name in hypothesis_model.hook_dict for hook_name in hook_names_for_patching]), \
    "All hook names for patching should be present in the hypothesis model."

  # For each hook name we need to decide what type of intervention we want to apply.
  options = InterventionType.get_available_interventions(residual_stream_mapper)

  # If max_interventions is greater than the total number of possible combinations, we will use all of them.
  # Otherwise, we will use a random sample of max_interventions.
  total_number_combinations = len(options) ** len(hook_names_for_patching)

  if max_interventions < total_number_combinations:
    indices = random.sample(range(total_number_combinations), max_interventions)
  else:
    indices = range(total_number_combinations)

  for index in indices:
    # build intervention for index
    intervention_types = np.base_repr(index, base=len(options)).zfill(len(hook_names_for_patching))
    intervention_types = [options[int(digit)] for digit in intervention_types]
    intervention = Intervention(hook_names_for_patching, intervention_types, residual_stream_mapper)
    yield intervention


def get_batched_intervention_data(
    clean_inputs: CaseDataset,
    corrupted_inputs: CaseDataset,
    base_model: HookedTransformer,
    hypothesis_model: HookedTransformer,
    residual_stream_mapper: ResidualStreamMapper | None = None,
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
    if residual_stream_mapper is not None:
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


def should_hook_name_be_skipped_due_to_filters(hook_name: str | None, hook_filters: List[str]) -> bool:
  if hook_filters is None:
    # No filters to apply
    return False

  if hook_name is None:
    # No hook name to apply the filters to
    return False

  return not any([filter in hook_name for filter in hook_filters])
