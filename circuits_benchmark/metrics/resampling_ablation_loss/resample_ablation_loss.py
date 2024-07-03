import gc
from dataclasses import dataclass
from typing import List, Dict, Optional

import numpy as np
import torch as t
from jaxtyping import Float, Int
from torch import Tensor
from transformer_lens import HookedTransformer, ActivationCache

from circuits_benchmark.benchmark.tracr_dataset import TracrDataset
from circuits_benchmark.metrics.resampling_ablation_loss.intervention import InterventionData
from circuits_benchmark.metrics.resampling_ablation_loss.resample_ablation_interventions import get_interventions
from circuits_benchmark.training.compression.activation_mapper.activation_mapper import ActivationMapper
from circuits_benchmark.training.compression.activation_mapper.multi_hook_activation_mapper import \
  MultiHookActivationMapper
from circuits_benchmark.utils.iit.iit_dataset_batch import IITDatasetBatch


@dataclass
class ResampleAblationLossOutput:
  loss: Float[Tensor, ""]
  variance_explained: Float[Tensor, ""]
  max_loss_per_node: Dict[str, Float[Tensor, ""]]
  mean_loss_per_node: Dict[str, Float[Tensor, ""]]
  interventions_per_node: Dict[str, int]
  intervened_nodes: List[str]


def get_resample_ablation_loss(data: IITDatasetBatch,
                               base_model: HookedTransformer,
                               hypothesis_model: HookedTransformer,
                               activation_mapper: MultiHookActivationMapper | ActivationMapper | None,
                               hook_filters: List[str] | None = None,
                               max_interventions: int = 10,
                               max_components: int = 1,
                               is_categorical: bool = False,
                               use_node_effect_diff: bool = False,
                               effect_diffs_by_node: Optional[Dict[str, float]] = None,
                               verbose: bool = False,
                               ) -> ResampleAblationLossOutput:
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

  # Prepare intervention data beforehand to avoid recomputing clean and corrupted activations on both models for each
  # intervention.
  clean_data, corrupted_data = data
  clean_inputs = clean_data[0]
  corrupted_inputs = corrupted_data[0]

  _, base_model_clean_cache = base_model.run_with_cache(clean_inputs)
  _, hypothesis_model_clean_cache = hypothesis_model.run_with_cache(clean_inputs)
  _, base_model_corrupted_cache = base_model.run_with_cache(corrupted_inputs)
  _, hypothesis_model_corrupted_cache = hypothesis_model.run_with_cache(corrupted_inputs)

  intervention_data = InterventionData(
    clean_inputs,
    base_model_corrupted_cache,
    hypothesis_model_corrupted_cache,
    base_model_clean_cache,
    hypothesis_model_clean_cache
  )

  # Calculate the variance of the base model logits.
  base_model_logits_variance = base_model(clean_inputs).var().item()

  # for each intervention, run both models, calculate MSE and add it to the losses.
  losses = []
  variance_explained = []
  max_loss_per_node = {}
  mean_loss_per_node = {}
  interventions_per_node = {}
  intervened_nodes = set()
  for intervention in get_interventions(base_model,
                                        hypothesis_model,
                                        hook_filters,
                                        activation_mapper,
                                        max_interventions,
                                        max_components,
                                        effect_diffs_by_node):

    if verbose:
      print(f"\nRunning intervention {intervention.node_intervention_types[0]} on node {intervention.node_names[0]}")

    for node_name in intervention.get_intervened_nodes():
      if node_name in interventions_per_node:
        interventions_per_node[node_name] += 1
      else:
        interventions_per_node[node_name] = 1

    with intervention.hooks(base_model, hypothesis_model, intervention_data):
      base_model_intervened_logits = base_model(clean_inputs)
      hypothesis_model_intervened_logits = hypothesis_model(clean_inputs)

    # The output has unspecified behavior for the BOS token, so we discard it on the loss calculation.
    base_model_intervened_logits = base_model_intervened_logits[:, 1:]
    hypothesis_model_intervened_logits = hypothesis_model_intervened_logits[:, 1:]

    if use_node_effect_diff:
      # We will compare the clean vs intervened logits of both models.
      base_model_clean_logits = base_model(clean_inputs)
      hypothesis_model_clean_logits = hypothesis_model(clean_inputs)

      # Remove BOS
      base_model_clean_logits = base_model_clean_logits[:, 1:]
      hypothesis_model_clean_logits = hypothesis_model_clean_logits[:, 1:]

      if is_categorical:
        # calculate log softmax and compare using mse loss
        # we want to know how much the distribution of probabilities changes for each model.
        base_model_clean_logits = t.nn.functional.log_softmax(base_model_clean_logits, dim=-1)
        hypothesis_model_clean_logits = t.nn.functional.log_softmax(hypothesis_model_clean_logits, dim=-1)
        base_model_intervened_logits = t.nn.functional.log_softmax(base_model_intervened_logits, dim=-1)
        hypothesis_model_intervened_logits = t.nn.functional.log_softmax(hypothesis_model_intervened_logits, dim=-1)

        base_model_effect = t.nn.functional.mse_loss(base_model_clean_logits, base_model_intervened_logits)
        hypothesis_model_effect = t.nn.functional.mse_loss(hypothesis_model_clean_logits, hypothesis_model_intervened_logits)

        intervention_loss = t.abs(base_model_effect - hypothesis_model_effect)

      else:
        base_model_effect = t.nn.functional.mse_loss(base_model_clean_logits, base_model_intervened_logits)
        hypothesis_model_effect = t.nn.functional.mse_loss(hypothesis_model_clean_logits, hypothesis_model_intervened_logits)
        intervention_loss = t.abs(base_model_effect - hypothesis_model_effect)

    else:
      # Just compare the intervened logits of both models.
      if is_categorical:
        # Use Cross Entropy loss for categorical outputs.
        flattened_intervened_logits: Float[
          Tensor, "batch*pos, vocab_out"] = hypothesis_model_intervened_logits.flatten(end_dim=-2)
        flattened_intervened_expected_labels: Int[Tensor, "batch*pos"] = base_model_intervened_logits.argmax(
          dim=-1).flatten()
        intervention_loss = t.nn.functional.cross_entropy(flattened_intervened_logits,
                                             flattened_intervened_expected_labels)
      else:
        # Use MSE loss for numerical outputs.
        intervention_loss = t.nn.functional.mse_loss(base_model_intervened_logits, hypothesis_model_intervened_logits)

    var_explained = 1 - intervention_loss / base_model_logits_variance

    losses.append(intervention_loss)
    variance_explained.append(var_explained)
    intervened_nodes.update(intervention.get_intervened_nodes())

    # store the max and mean loss per hook for the intervention.
    for node_name in intervention.get_intervened_nodes():
      if node_name in max_loss_per_node:
        max_loss_per_node[node_name] = max(max_loss_per_node[node_name], intervention_loss.item())
      else:
        max_loss_per_node[node_name] = intervention_loss

      if node_name in mean_loss_per_node:
        mean_loss_per_node[node_name].append(intervention_loss)
      else:
        mean_loss_per_node[node_name] = [intervention_loss]

  for node_name in mean_loss_per_node:
    mean_loss_per_node[node_name] = t.stack(mean_loss_per_node[node_name], dim=0).mean()

  return ResampleAblationLossOutput(
    loss=t.stack(losses).mean(),
    variance_explained=t.stack(variance_explained).mean(),
    max_loss_per_node=max_loss_per_node,
    mean_loss_per_node=mean_loss_per_node,
    interventions_per_node=interventions_per_node,
    intervened_nodes=list(intervened_nodes)
  )


def get_batched_intervention_data(
    clean_data: TracrDataset,
    corrupted_data: TracrDataset,
    base_model: HookedTransformer,
    hypothesis_model: HookedTransformer,
    activation_mapper: MultiHookActivationMapper | ActivationMapper | None = None,
    batch_size: int = 2048,
    hypothesis_model_corrupted_cache: ActivationCache | None = None,
) -> List[InterventionData]:
  data = []
  batches_count = 0

  for clean_data_batch, corrupted_data_batch in zip(clean_data.make_loader(batch_size),
                                                        corrupted_data.make_loader(batch_size)):
    batches_count += 1
    clean_inputs_batch = clean_data_batch[0]
    corrupted_inputs_batch = corrupted_data_batch[0]

    # Run the corrupted inputs on both models and save the activation caches.
    _, base_model_corrupted_cache = base_model.run_with_cache(corrupted_inputs_batch)

    if hypothesis_model_corrupted_cache is None:
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

  assert hypothesis_model_corrupted_cache is None or batches_count == 1, \
    "The hypothesis model corrupted cache optimization argument should only when there is a single batch."

  return data
