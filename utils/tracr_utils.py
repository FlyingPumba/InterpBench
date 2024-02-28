from typing import List

from tracr.craft import transformers
from tracr.craft.transformers import SeriesWithResiduals


def get_relevant_component_names(craft_model: SeriesWithResiduals) -> List[str]:
  """Return the names of the MLP and Attention modules that are used by the craft model."""
  component_names: List[str] = []

  candidate_component_names = []
  for layer in range(len(craft_model.blocks)):
    candidate_component_names.append(f"blocks.{layer}.attn")
    candidate_component_names.append(f"blocks.{layer}.mlp")
  candidate_component_names = iter(candidate_component_names)

  for module in craft_model.blocks:
    if isinstance(module, transformers.MLP):
      layer_type = "mlp"
    else:
      layer_type = "attn"
    # Find next layer with the necessary type. Modules in-between, that are not
    # added to component_names will be disabled later by setting all weights to 0.
    component_name = next(candidate_component_names)
    while layer_type not in component_name:
      component_name = next(candidate_component_names)
    component_names.append(component_name)

  return component_names


def get_output_space_basis_name(block):
  output_basis = None
  if isinstance(block, transformers.MLP):
    output_basis = block.snd.output_space.basis
  elif isinstance(block, transformers.MultiAttentionHead):
    assert len(block.sub_blocks) == 1, "Only one sub block is supported."
    output_basis = block.sub_blocks[0].w_ov.output_space.basis
  elif isinstance(block, transformers.AttentionHead):
    output_basis = block.w_ov.output_space.basis
  else:
    raise ValueError(f"Unsupported block type: {block}")

  assert len(output_basis) == 1, "Only one basis per output space is supported."
  return output_basis[0].name
