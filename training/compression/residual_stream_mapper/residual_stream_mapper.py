from abc import ABC, abstractmethod

from jaxtyping import Float
from torch import Tensor


class ResidualStreamMapper(ABC):
  """Maps the residual stream to/from a lower dimensional space."""

  @abstractmethod
  def compress(
      self,
      residual_stream: Float[Tensor, "batch d_model"]
  ) -> Float[Tensor, "batch d_model_compressed"]:
    raise NotImplementedError

  @abstractmethod
  def decompress(
      self,
      compressed_residual_stream: Float[Tensor, "batch d_model_compressed"]
  ) -> Float[Tensor, "batch d_model"]:
    raise NotImplementedError
