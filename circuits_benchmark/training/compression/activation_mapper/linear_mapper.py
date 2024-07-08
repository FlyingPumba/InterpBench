from jaxtyping import Float
from torch import Tensor
from torch.nn import Linear

from circuits_benchmark.training.compression.activation_mapper.activation_mapper import ActivationMapper


class LinearMapper(ActivationMapper):
  """Maps the residual stream to/from a lower dimensional space using a linear layer."""

  def __init__(self, compression_matrix: Linear):
    self.compression_matrix = compression_matrix

  def compress(
      self,
      residual_stream: Float[Tensor, "batch d_model"]
  ) -> Float[Tensor, "batch d_model_compressed"]:
    return residual_stream @ self.compression_matrix.weight

  def decompress(
      self,
      compressed_residual_stream: Float[Tensor, "batch d_model_compressed"]
  ) -> Float[Tensor, "batch d_model"]:
    return compressed_residual_stream @ self.compression_matrix.weight.T