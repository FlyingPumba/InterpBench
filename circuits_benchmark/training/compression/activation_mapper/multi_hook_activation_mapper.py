import re

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
        hook_name: str,
        head_index: int = None,
    ) -> Float[Tensor, "batch pos d_model_compressed"] | Float[Tensor, "batch pos head_index d_head_compressed"]:
        """Compresses an activation."""
        hook_mapper = self.find_mapper(hook_name, head_index)
        assert hook_mapper is not None, f"Could not find a mapper for hook name {hook_name} and head index {head_index}"
        return hook_mapper.compress(activation)

    def decompress(
        self,
        compressed_activation: Float[Tensor, "batch pos d_model_compressed"] |
                               Float[Tensor, "batch pos head_index d_head_compressed"],
        hook_name: str,
        head_index: int = None,
    ) -> Float[Tensor, "batch pos d_model"] | Float[Tensor, "batch pos head_index d_head"]:
        """Decompresses a compressed activation."""
        hook_mapper = self.find_mapper(hook_name, head_index)
        assert hook_mapper is not None, f"Could not find a mapper for hook name {hook_name} and head index {head_index}"
        return hook_mapper.decompress(compressed_activation)

    def find_mapper(self, hook_name: str, head_index: int = None) -> ActivationMapper | None:
        """Finds the mapper for the given hook name."""
        full_node_name = f"{hook_name}[{head_index}]" if head_index is not None else hook_name

        for mapper_key, mapper in self.mappers_dict.items():
            if "[" in mapper_key and head_index is None:
                # remove the head index from the mapper_key
                mapper_key = mapper_key.split("[")[0]

            # replace square brackets and dots with escaped versions
            mapper_key = mapper_key.replace("[", "\\[").replace("]", "\\]")

            # convert the mapper_key to a regex pattern
            regex = re.compile(f"^{mapper_key}$")
            if regex.match(full_node_name):
                return mapper

        return None

    def supports_hook(self, hook_name: str) -> bool:
        """Returns whether the multi activation mapper supports the given hook name."""
        return self.find_mapper(hook_name) is not None
