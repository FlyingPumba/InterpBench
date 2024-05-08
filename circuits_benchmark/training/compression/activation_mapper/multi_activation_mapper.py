import re

from jaxtyping import Float
from torch import Tensor

from circuits_benchmark.training.compression.activation_mapper.activation_mapper import ActivationMapper


class MultiActivationMapper(object):
  """Maps multiple activations from different hook names to/from a lower dimension space."""
  def __init__(self, mappers_dict: dict[str, ActivationMapper]):
    self.mappers_dict = mappers_dict

  def compress(
      self,
      activation: Float[Tensor, "batch pos d_model"] | Float[Tensor, "batch pos d_head"],
      hook_name: str
  ) -> Float[Tensor, "batch pos d_model_compressed"] | Float[Tensor, "batch pos d_head_compressed"]:
    """Compresses an activation."""
    mapper = self.find_mapper(hook_name)
    return mapper.compress(activation)

  def decompress(
      self,
      compressed_activation: Float[Tensor, "batch pos d_model_compressed"] | Float[Tensor, "batch pos d_head_compressed"],
      hook_name: str,
  ) -> Float[Tensor, "batch pos d_model"] | Float[Tensor, "batch pos d_head"]:
    """Decompresses a compressed activation."""
    mapper = self.find_mapper(hook_name)
    return mapper.decompress(compressed_activation)

  def find_mapper(self, hook_name: str) -> ActivationMapper:
    """Finds the mapper for the given hook name."""
    # Find the mapper for which the key is a valid regex that matches the hook_name
    for mapper_key, mapper in self.mappers_dict.items():
      # convert the ae_key to a regex pattern
      regex = re.compile(f"^{mapper_key}$")
      if regex.match(hook_name):
        return mapper

    raise ValueError(f"Could not find a mapper for hook_name: {hook_name}")

  def supports_hook(self, hook_name: str):
    """Returns whether the multi activation mapper supports the given hook name."""
    for mapper_key in self.mappers_dict.keys():
      regex = re.compile(f"^{mapper_key}$")
      if regex.match(hook_name):
        return True
    return False