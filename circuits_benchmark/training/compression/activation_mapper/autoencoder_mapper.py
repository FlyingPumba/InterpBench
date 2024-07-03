from jaxtyping import Float
from torch import Tensor

from circuits_benchmark.training.compression.activation_mapper.activation_mapper import ActivationMapper
from circuits_benchmark.training.compression.autencoder import AutoEncoder


class AutoEncoderMapper(ActivationMapper):
  """Maps the residual stream to/from a lower dimensional space using an autoencoder."""

  def __init__(self, autoencoder: AutoEncoder):
    self.autoencoder = autoencoder

  def compress(
      self,
      residual_stream: Float[Tensor, "batch d_model"]
  ) -> Float[Tensor, "batch d_model_compressed"]:
    return self.autoencoder.encoder(residual_stream)

  def decompress(
      self,
      compressed_residual_stream: Float[Tensor, "batch d_model_compressed"]
  ) -> Float[Tensor, "batch d_model"]:
    return self.autoencoder.decoder(compressed_residual_stream)