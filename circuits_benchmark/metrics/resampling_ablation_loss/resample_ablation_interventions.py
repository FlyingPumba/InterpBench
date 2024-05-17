import random
from typing import List, Generator

from transformer_lens import HookedTransformer

from circuits_benchmark.metrics.resampling_ablation_loss.intervention import Intervention
from circuits_benchmark.metrics.resampling_ablation_loss.intervention_type import InterventionType
from circuits_benchmark.training.compression.activation_mapper.activation_mapper import ActivationMapper
from circuits_benchmark.training.compression.activation_mapper.multi_hook_activation_mapper import \
  MultiHookActivationMapper


def get_interventions(
    base_model: HookedTransformer,
    hypothesis_model: HookedTransformer,
    hook_filters: List[str],
    activation_mapper: MultiHookActivationMapper | ActivationMapper | None = None,
    max_interventions: int = 10,
    max_components: int = 1) -> Generator[Intervention, None, None]:
  """Builds the different combinations for possible interventions on the base and hypothesis models."""
  hook_names: List[str | None] = list(base_model.hook_dict.keys())
  hook_names_for_patching = [name for name in hook_names
                             if not should_hook_name_be_skipped_due_to_filters(name, hook_filters)]

  # assert all hook names for patching are also present in the hypothesis model
  assert all([hook_name in hypothesis_model.hook_dict for hook_name in hook_names_for_patching]), \
    "All hook names for patching should be present in the hypothesis model."

  # add attention heads to attention hook names that need it
  node_names_for_patching = []
  attn_head_hooks = [
    "attn.hook_result",
    "attn.hook_z",
    "attn.hook_attn_scores",
    "attn.hook_pattern",
    "attn.hook_result",
  ]
  for letter in "qkv":
    attn_head_hooks.append(f"attn.hook_{letter}")
    attn_head_hooks.append(f"hook_{letter}_input")
  for hook_name in hook_names_for_patching[:]:
    if any([hook_name.endswith(attn_head_hook) for attn_head_hook in attn_head_hooks]):
      # add attention head version of hook name
      for head in range(base_model.cfg.n_heads):
        node_names_for_patching.append(f"{hook_name}[{head}]")
    else:
      node_names_for_patching.append(hook_name)

  # For each hook name we need to decide what type of intervention we want to apply.
  options = InterventionType.get_available_interventions(activation_mapper)
  options.remove(InterventionType.NO_INTERVENTION)

  for _ in range(max_interventions):
    # choose max_components_to_intervene (no replacement) out of the node_names_for_patching
    node_names_to_intervene = random.sample(node_names_for_patching, max_components)

    # randomly choose the intervention type for each hook name
    intervention_types = [random.choice(options) for _ in range(len(node_names_to_intervene))]

    intervention = Intervention(node_names_to_intervene, intervention_types, activation_mapper)
    yield intervention


def should_hook_name_be_skipped_due_to_filters(hook_name: str | None, hook_filters: List[str]) -> bool:
  if hook_filters is None:
    # No filters to apply
    return False

  if hook_name is None:
    # No hook name to apply the filters to
    return False

  return not any([filter in hook_name for filter in hook_filters])
