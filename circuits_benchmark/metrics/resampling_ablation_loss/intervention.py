from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from typing import List

from jaxtyping import Float
from torch import Tensor
from transformer_lens import ActivationCache, HookedTransformer
from transformer_lens.hook_points import HookPoint

from circuits_benchmark.metrics.resampling_ablation_loss.intervention_type import InterventionType
from circuits_benchmark.training.compression.activation_mapper.activation_mapper import ActivationMapper
from circuits_benchmark.training.compression.activation_mapper.multi_hook_activation_mapper import MultiHookActivationMapper
from circuits_benchmark.transformers.hooked_tracr_transformer import HookedTracrTransformerBatchInput


def regular_intervention_hook_fn(
    activation: Float[Tensor, "batch seq_len d_model"],
    hook: HookPoint,
    corrupted_cache: ActivationCache = None,
    head_index: int = None
):
  """This hook just replaces the output with a corrupted output."""
  if head_index is None:
    return corrupted_cache[hook.name]
  else:
    activation[:, :, head_index] = corrupted_cache[hook.name][:, :, head_index]
    return activation


def compression_intervention_hook_fn(
    activation: Float[Tensor, "batch seq_len d_model"],
    hook: HookPoint,
    corrupted_cache: ActivationCache = None,
    head_index: int = None,
    activation_mapper: MultiHookActivationMapper | ActivationMapper | None = None
):
  """This hook replaces the output with a corrupted output passed through the compressor."""
  if activation_mapper is None:
    raise ValueError("Compression intervention requires an activation mapper.")

  if head_index is None:
    if isinstance(activation_mapper, ActivationMapper):
      return activation_mapper.compress(corrupted_cache[hook.name])
    else:
      return activation_mapper.compress(corrupted_cache[hook.name], hook.name)
  else:
    if isinstance(activation_mapper, ActivationMapper):
      activation[:, :, head_index] = activation_mapper.compress(corrupted_cache[hook.name][:, :, head_index])
    else:
      activation[:, :, head_index] = activation_mapper.compress(corrupted_cache[hook.name][:, :, head_index], hook.name, head_index)
    return activation


def decompression_intervention_hook_fn(
    activation: Float[Tensor, "batch seq_len d_model"],
    hook: HookPoint,
    corrupted_cache: ActivationCache = None,
    head_index: int = None,
    activation_mapper: MultiHookActivationMapper | ActivationMapper | None = None
):
  """This hook replaces the output with a corrupted output passed through the decompressor."""
  if activation_mapper is None:
    raise ValueError("Decompression intervention requires an activation mapper.")

  if head_index is None:
    if isinstance(activation_mapper, ActivationMapper):
      return activation_mapper.decompress(corrupted_cache[hook.name])
    else:
      return activation_mapper.decompress(corrupted_cache[hook.name], hook.name)
  else:
    if isinstance(activation_mapper, ActivationMapper):
      activation[:, :, head_index] = activation_mapper.decompress(corrupted_cache[hook.name][:, :, head_index])
    else:
      activation[:, :, head_index] = activation_mapper.decompress(corrupted_cache[hook.name][:, :, head_index], hook.name, head_index)
    return activation


@dataclass
class InterventionData:
  clean_inputs: HookedTracrTransformerBatchInput
  base_model_corrupted_cache: ActivationCache
  hypothesis_model_corrupted_cache: ActivationCache
  base_model_clean_cache: ActivationCache | None
  hypothesis_model_clean_cache: ActivationCache | None


class Intervention(object):
  def __init__(self,
               node_names: List[str],
               node_intervention_types: List[InterventionType],
               activation_mapper: MultiHookActivationMapper | ActivationMapper | None = None):
    self.node_names = node_names
    self.node_intervention_types = node_intervention_types
    self.activation_mapper = activation_mapper

    assert len(node_names) == len(node_intervention_types), \
      "node_names and node_intervention_types should have the same length."

    # Assert there are no interventions that require a residual stream mapper if it is not provided.
    if activation_mapper is None:
      for intervention_type in node_intervention_types:
        assert intervention_type not in [InterventionType.CORRUPTED_COMPRESSION,
                                         InterventionType.CORRUPTED_DECOMPRESSION,
                                         InterventionType.CLEAN_COMPRESSION,
                                         InterventionType.CLEAN_DECOMPRESSION], \
          "ResidualStreamMapper is not provided, so interventions that require it are not allowed."

    self.has_clean_interventions = any([intervention_type in [InterventionType.CLEAN_COMPRESSION,
                                                              InterventionType.CLEAN_DECOMPRESSION]
                                        for intervention_type in node_intervention_types])

  def get_intervened_nodes(self):
    """Returns the hook names that will be intervened."""
    return [node_name for node_name, intervention_type in zip(self.node_names, self.node_intervention_types)
            if intervention_type != InterventionType.NO_INTERVENTION]

  def get_intervened_hook_names(self):
    """Returns the hook names that will be intervened."""
    return [node_name.split("[")[0] for node_name, intervention_type in zip(self.node_names, self.node_intervention_types)
            if intervention_type != InterventionType.NO_INTERVENTION]

  def get_params_affected_by_interventions(self):
    affected_params = set()

    for hook_name in self.get_intervened_hook_names():
      if hook_name == "hook_embed":
        affected_params.add("embed.W_E")
      elif hook_name == "hook_pos_embed":
        affected_params.add("pos_embed.W_pos")
      elif hook_name.endswith("hook_result"):  # e.g., blocks.3.attn.hook_result
        layer = int(hook_name.split(".")[1])
        # add all params involved in attention head for this layer
        for param_name in [f"blocks.{layer}.attn.W_{param}" for param in ["Q", "O", "K", "V"]] + \
                          [f"blocks.{layer}.attn.b_{param}" for param in ["Q", "O", "K", "V"]]:
          affected_params.add(param_name)
      elif hook_name.endswith("hook_mlp_out"):  # e.g., blocks.1.hook_mlp_out
        layer = int(hook_name.split(".")[1])
        # add all params involved in MLP for this layer
        for param_name in [f"blocks.{layer}.mlp.W_{param}" for param in ["in", "out"]] + \
                          [f"blocks.{layer}.mlp.b_{param}" for param in ["in", "out"]]:
          affected_params.add(param_name)
      else:
        raise NotImplementedError(f"Hook name {hook_name} is not supported.")

    return affected_params


  @contextmanager
  def hooks(self,
            base_model: HookedTransformer,
            hypothesis_model: HookedTransformer,
            intervention_data: InterventionData):
    base_model_corrupted_cache = intervention_data.base_model_corrupted_cache
    hypothesis_model_corrupted_cache = intervention_data.hypothesis_model_corrupted_cache
    base_model_clean_cache = intervention_data.base_model_clean_cache
    hypothesis_model_clean_cache = intervention_data.hypothesis_model_clean_cache

    if self.has_clean_interventions:
      assert base_model_clean_cache is not None and hypothesis_model_clean_cache is not None, \
        "Clean caches are required for clean interventions."

    base_model_hooks = []
    hypothesis_model_hooks = []

    for node_name, intervention_type in zip(self.node_names, self.node_intervention_types):
      head_index = None
      if "[" in node_name:
        hook_name, head_index = node_name.split("[")
        head_index = int(head_index[:-1])
      else:
        hook_name = node_name

      if intervention_type == InterventionType.REGULAR_CORRUPTED:
        base_model_hooks.append((hook_name, partial(regular_intervention_hook_fn,
                                                    corrupted_cache=base_model_corrupted_cache,
                                                    head_index=head_index)))
        hypothesis_model_hooks.append((hook_name, partial(regular_intervention_hook_fn,
                                                          corrupted_cache=hypothesis_model_corrupted_cache,
                                                          head_index=head_index)))

      elif intervention_type == InterventionType.CORRUPTED_COMPRESSION:
        base_model_hooks.append((hook_name, partial(regular_intervention_hook_fn,
                                                    corrupted_cache=base_model_corrupted_cache,
                                                    head_index=head_index)))
        hypothesis_model_hooks.append((hook_name, partial(compression_intervention_hook_fn,
                                                          corrupted_cache=base_model_corrupted_cache,
                                                          activation_mapper=self.activation_mapper,
                                                          head_index=head_index)))

      elif intervention_type == InterventionType.CORRUPTED_DECOMPRESSION:
        base_model_hooks.append((hook_name, partial(decompression_intervention_hook_fn,
                                                    corrupted_cache=hypothesis_model_corrupted_cache,
                                                    activation_mapper=self.activation_mapper,
                                                    head_index=head_index)))
        hypothesis_model_hooks.append((hook_name, partial(regular_intervention_hook_fn,
                                                          corrupted_cache=hypothesis_model_corrupted_cache,
                                                          head_index=head_index)))

      elif intervention_type == InterventionType.CLEAN_COMPRESSION:
        hypothesis_model_hooks.append((hook_name, partial(compression_intervention_hook_fn,
                                                          corrupted_cache=base_model_clean_cache,
                                                          activation_mapper=self.activation_mapper,
                                                          head_index=head_index)))

      elif intervention_type == InterventionType.CLEAN_DECOMPRESSION:
        base_model_hooks.append((hook_name, partial(decompression_intervention_hook_fn,
                                                    corrupted_cache=hypothesis_model_clean_cache,
                                                    activation_mapper=self.activation_mapper,
                                                    head_index=head_index)))

      elif intervention_type == InterventionType.NO_INTERVENTION:
        # No hooks to add
        pass

      else:
        raise ValueError(f"Intervention type {intervention_type} is not supported.")

    with base_model.hooks(base_model_hooks):
      with hypothesis_model.hooks(hypothesis_model_hooks):
        yield self
