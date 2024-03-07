import random
from typing import List, Generator

import numpy as np
from transformer_lens import HookedTransformer

from circuits_benchmark.metrics.resampling_ablation_loss.intervention import Intervention
from circuits_benchmark.metrics.resampling_ablation_loss.intervention_type import InterventionType
from circuits_benchmark.training.compression.residual_stream_mapper.residual_stream_mapper import ResidualStreamMapper


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


def should_hook_name_be_skipped_due_to_filters(hook_name: str | None, hook_filters: List[str]) -> bool:
  if hook_filters is None:
    # No filters to apply
    return False

  if hook_name is None:
    # No hook name to apply the filters to
    return False

  return not any([filter in hook_name for filter in hook_filters])
