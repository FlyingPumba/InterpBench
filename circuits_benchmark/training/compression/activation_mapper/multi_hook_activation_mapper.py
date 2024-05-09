import re
from typing import List

import torch as t
from jaxtyping import Float
from torch import Tensor

from circuits_benchmark.training.compression.activation_mapper.activation_mapper import ActivationMapper


class MultiHookActivationMapper(object):
  """Maps multiple activations from different hook names to/from a lower dimension space."""
  def __init__(self, mappers_dict: dict[str, ActivationMapper]):
    self.mappers_dict = mappers_dict

  def compress(
      self,
      activation: Float[Tensor, "batch pos d_model"] | Float[Tensor, "batch pos head_index d_head"],
      hook_name: str
  ) -> Float[Tensor, "batch pos d_model_compressed"] | Float[Tensor, "batch pos head_index d_head_compressed"]:
    """Compresses an activation."""
    hook_mappers = self.find_mappers(hook_name)

    if activation.ndim == 3:
      assert len(hook_mappers) == 1, f"Number of mappers ({len(hook_mappers)}) for d_model activations must be equal to 1"
      return hook_mappers[0].compress(activation)
    elif activation.ndim == 4:
      n_heads = activation.shape[-2]
      assert len(hook_mappers) == n_heads, \
        f"Number of mappers ({len(hook_mappers)}) must be equal to the number of heads ({n_heads})"
      compressed_activations = [hook_mappers[i].compress(activation[:, :, i, :]) for i in range(n_heads)]
      return t.stack(compressed_activations, dim=2)

  def decompress(
      self,
      compressed_activation: Float[Tensor, "batch pos d_model_compressed"] |
                             Float[Tensor, "batch pos head_index d_head_compressed"],
      hook_name: str,
  ) -> Float[Tensor, "batch pos d_model"] | Float[Tensor, "batch pos head_index d_head"]:
    """Decompresses a compressed activation."""
    hook_mappers = self.find_mappers(hook_name)

    if compressed_activation.ndim == 3:
      assert len(hook_mappers) == 1, f"Number of mappers ({len(hook_mappers)}) for d_model activations must be equal to 1"
      return hook_mappers[0].decompress(compressed_activation)
    elif compressed_activation.ndim == 4:
      n_heads = compressed_activation.shape[-2]
      assert len(hook_mappers) == n_heads, \
        f"Number of mappers ({len(hook_mappers)}) must be equal to the number of heads ({n_heads})"
      decompressed_activations = [hook_mappers[i].decompress(compressed_activation[:, :, i, :]) for i in range(n_heads)]
      return t.stack(decompressed_activations, dim=2)

  def find_mappers(self, hook_name: str) -> List[ActivationMapper]:
    """Finds the mapper for the given hook name."""
    mappers = {}
    for mapper_key, mapper in self.mappers_dict.items():
      # convert the mapper_key to a regex pattern
      key_for_regex = mapper_key.split("[")[0]
      regex = re.compile(f"^{key_for_regex}$")
      if regex.match(hook_name):
        mappers[mapper_key] = mapper

    # return values in mappers, ordered by key
    return [mappers[key] for key in sorted(mappers.keys())]

  def supports_hook(self, hook_name: str):
    """Returns whether the multi activation mapper supports the given hook name."""
    return len(self.find_mappers(hook_name)) > 0