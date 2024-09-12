from abc import ABC, abstractmethod

from jaxtyping import Float
from torch import Tensor


class ActivationMapper(ABC):
    """Maps activations to/from a lower dimensional space."""

    @abstractmethod
    def compress(
        self,
        activation: Float[Tensor, "batch pos d_model"] | Float[Tensor, "batch pos d_head"]
    ) -> Float[Tensor, "batch pos d_model_compressed"] | Float[Tensor, "batch pos d_head_comprssed"]:
        """Compresses an activation."""
        raise NotImplementedError

    @abstractmethod
    def decompress(
        self,
        compressed_activation: Float[Tensor, "batch pos d_model_compressed"] | Float[
            Tensor, "batch pos d_head_comprssed"]
    ) -> Float[Tensor, "batch pos d_model"] | Float[Tensor, "batch pos d_head"]:
        """Decompresses a compressed activation."""
        raise NotImplementedError
