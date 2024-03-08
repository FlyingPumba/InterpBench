from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from typing import List

from jaxtyping import Float
from torch import Tensor
from transformer_lens import ActivationCache, HookedTransformer
from transformer_lens.hook_points import HookPoint

from circuits_benchmark.metrics.resampling_ablation_loss.intervention_type import InterventionType
from circuits_benchmark.training.compression.residual_stream_mapper.residual_stream_mapper import ResidualStreamMapper
from circuits_benchmark.transformers.hooked_tracr_transformer import HookedTracrTransformerBatchInput


def regular_intervention_hook_fn(
    residual_stream: Float[Tensor, "batch seq_len d_model"],
    hook: HookPoint,
    corrupted_cache: ActivationCache = None
):
  """This hook just replaces the output with a corrupted output."""
  return corrupted_cache[hook.name]


def compression_intervention_hook_fn(
    residual_stream: Float[Tensor, "batch seq_len d_model"],
    hook: HookPoint,
    corrupted_cache: ActivationCache = None,
    residual_stream_mapper: ResidualStreamMapper | None = None
):
  """This hook replaces the output with a corrupted output passed through the compressor."""
  return residual_stream_mapper.compress(corrupted_cache[hook.name])


def decompression_intervention_hook_fn(
    residual_stream: Float[Tensor, "batch seq_len d_model"],
    hook: HookPoint,
    corrupted_cache: ActivationCache = None,
    residual_stream_mapper: ResidualStreamMapper | None = None
):
  """This hook replaces the output with a corrupted output passed through the decompressor."""
  return residual_stream_mapper.decompress(corrupted_cache[hook.name])


@dataclass
class InterventionData:
  clean_inputs: HookedTracrTransformerBatchInput
  base_model_corrupted_cache: ActivationCache
  hypothesis_model_corrupted_cache: ActivationCache
  base_model_clean_cache: ActivationCache | None
  hypothesis_model_clean_cache: ActivationCache | None


class Intervention(object):
  def __init__(self,
               hook_names: List[str],
               hook_intervention_types: List[InterventionType],
               residual_stream_mapper: ResidualStreamMapper | None = None):
    self.hook_names = hook_names
    self.hook_intervention_types = hook_intervention_types
    self.residual_stream_mapper = residual_stream_mapper

    assert len(hook_names) == len(hook_intervention_types), \
      "hook_names and hook_intervention_types should have the same length."

    # Assert there are no interventions that require a residual stream mapper if it is not provided.
    if residual_stream_mapper is None:
      for intervention_type in hook_intervention_types:
        assert intervention_type not in [InterventionType.CORRUPTED_COMPRESSION,
                                         InterventionType.CORRUPTED_DECOMPRESSION,
                                         InterventionType.CLEAN_COMPRESSION,
                                         InterventionType.CLEAN_DECOMPRESSION], \
          "ResidualStreamMapper is not provided, so interventions that require it are not allowed."

    self.has_clean_interventions = any([intervention_type in [InterventionType.CLEAN_COMPRESSION,
                                                              InterventionType.CLEAN_DECOMPRESSION]
                                        for intervention_type in hook_intervention_types])

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

    for hook_name, intervention_type in zip(self.hook_names, self.hook_intervention_types):
      if intervention_type == InterventionType.REGULAR_CORRUPTED:
        base_model_hooks.append((hook_name, partial(regular_intervention_hook_fn,
                                                    corrupted_cache=base_model_corrupted_cache)))
        hypothesis_model_hooks.append((hook_name, partial(regular_intervention_hook_fn,
                                                          corrupted_cache=hypothesis_model_corrupted_cache)))

      elif intervention_type == InterventionType.CORRUPTED_COMPRESSION:
        base_model_hooks.append((hook_name, partial(regular_intervention_hook_fn,
                                                    corrupted_cache=base_model_corrupted_cache)))
        hypothesis_model_hooks.append((hook_name, partial(compression_intervention_hook_fn,
                                                          corrupted_cache=base_model_corrupted_cache,
                                                          residual_stream_mapper=self.residual_stream_mapper)))

      elif intervention_type == InterventionType.CORRUPTED_DECOMPRESSION:
        base_model_hooks.append((hook_name, partial(decompression_intervention_hook_fn,
                                                    corrupted_cache=hypothesis_model_corrupted_cache,
                                                    residual_stream_mapper=self.residual_stream_mapper)))
        hypothesis_model_hooks.append((hook_name, partial(regular_intervention_hook_fn,
                                                          corrupted_cache=hypothesis_model_corrupted_cache)))

      elif intervention_type == InterventionType.CLEAN_COMPRESSION:
        hypothesis_model_hooks.append((hook_name, partial(compression_intervention_hook_fn,
                                                          corrupted_cache=base_model_clean_cache,
                                                          residual_stream_mapper=self.residual_stream_mapper)))

      elif intervention_type == InterventionType.CLEAN_DECOMPRESSION:
        base_model_hooks.append((hook_name, partial(decompression_intervention_hook_fn,
                                                    corrupted_cache=hypothesis_model_clean_cache,
                                                    residual_stream_mapper=self.residual_stream_mapper)))

      elif intervention_type == InterventionType.NO_INTERVENTION:
        # No hooks to add
        pass

      else:
        raise ValueError(f"Intervention type {intervention_type} is not supported.")

    with base_model.hooks(base_model_hooks):
      with hypothesis_model.hooks(hypothesis_model_hooks):
        yield self
