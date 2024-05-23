import math

import torch as t


def small_init_init_method(dim):
  """Fills the input Tensor with values according to the method described in Transformers without Tears: Improving
  the Normalization of Self-Attention - Nguyen, T. & Salazar, J. (2010), using a normal distribution."""
  std = math.sqrt(2 / (5 * dim))

  def init_(tensor):
    return t.nn.init.normal_(tensor, mean=0.0, std=std)

  return init_


def wang_init_method(n_layers, dim):
  std = 2 / n_layers / math.sqrt(dim)  # Equivalent to (2 / n_layers) * (1 / math.sqrt(dim))

  def init_(tensor):
    return t.nn.init.normal_(tensor, mean=0.0, std=std)

  return init_


def kaiming_uniform_and_normal_for_biases():
  def init_(tensor):
    if len(tensor.shape) > 1:
      return t.nn.init.kaiming_uniform_(tensor)
    else:
      # Biases are initialized with a normal distribution
      return t.nn.init.normal_(tensor, std=0.02)

  return init_