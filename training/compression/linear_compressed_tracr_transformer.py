from __future__ import annotations

import os
import typing
from typing import List, Tuple, Callable, Literal

import numpy as np
import torch as t
from jaxtyping import Float
from torch import nn, Tensor
from torch.nn import Linear
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint

from utils.hooked_tracr_transformer import HookedTracrTransformer

LinearCompressedTracrTransformerInitialization = Literal["orthogonal", "linear"]
linear_compression_initialization_options = list(typing.get_args(LinearCompressedTracrTransformerInitialization))


class LinearCompressedTracrTransformer(HookedTracrTransformer):
  """ A transformer model with a linearly compressed residual stream.
  To train the model, we multiply with a matrix W when reading from the residual stream, and with W^T when writing to
  the residual stream.
  """

  def __init__(self,
               tl_model: HookedTracrTransformer,
               residual_stream_compression_size: int,
               linear_compression_initialization: LinearCompressedTracrTransformerInitialization = "linear",
               device: t.device = t.device("cuda") if t.cuda.is_available() else t.device("cpu"),
               *args, **kwargs):
    super().__init__(
      cfg=tl_model.cfg,
      tracr_input_encoder=tl_model.tracr_input_encoder,
      tracr_output_encoder=tl_model.tracr_output_encoder,
      residual_stream_labels=tl_model.residual_stream_labels,
      device=device,
      *args, **kwargs)
    self.original_residual_stream_size = tl_model.cfg.d_model
    self.residual_stream_compression_size = residual_stream_compression_size
    self.device = device

    self.load_weights_from_tl_model(tl_model)

    # [to_size, from_size]
    self.W_compress: Linear = nn.Linear(self.original_residual_stream_size,
                                        self.residual_stream_compression_size,
                                        device=self.device,
                                        bias=False)

    if linear_compression_initialization == "orthogonal":
      # The (semi) orthogonal matrix is useful for our setup because the transpose is exactly the inverse, and we will
      # use the same matrix for reading and writing from/to the residual stream.
      nn.init.orthogonal_(self.W_compress.weight)

    self.reset_hooks(including_permanent=True)
    self.add_linear_compression_hooks()

  def load_weights_from_tl_model(self, tl_model: HookedTransformer):
    """ Load the weights from a HookedTracrTransformer. We use strict=False because the tl_model will probably not have
    weights for the linear compression matrix. We then freeze all weights.
    """
    self.load_state_dict(tl_model.state_dict(), strict=False)

    for param in self.parameters():
      param.requires_grad = False

  def add_linear_compression_hooks(self):
    hooks = self.build_linear_compression_hooks()
    for hook_name, hook_function in hooks:
      self.add_hook(hook_name, hook_function, is_permanent=True)

  def build_linear_compression_hooks(self) -> List[Tuple[str, Callable]]:
    hooks = []

    write_to_compressed_residual = lambda x: x @ self.W_compress.weight.T
    read_from_compressed_residual = lambda x: x @ self.W_compress.weight

    def write_to_resid_hook_function(
        residual_stream: Float[Tensor, "batch seq_len d_model"],
        hook: HookPoint
    ) -> Float[Tensor, "batch seq_len d_model_compressed"]:
      return write_to_compressed_residual(residual_stream)

    def read_from_resid_hook_function(
        residual_stream: Float[Tensor, "batch seq_len n_head d_model_compressed"],
        hook: HookPoint
    ) -> Float[Tensor, "batch seq_len n_head d_model"]:
      return read_from_compressed_residual(residual_stream)

    # Add hooks for Attention heads and MLPs
    for hook_name in self.hook_dict.keys():
      if "hook_k_input" in hook_name or "hook_q_input" in hook_name or "hook_v_input" in hook_name:
        # Attention head matrices read directly from the residual stream
        hooks.append((hook_name, read_from_resid_hook_function))
      elif "hook_attn_out" in hook_name:
        # the output of attention heads is written directly to the residual stream
        hooks.append((hook_name, write_to_resid_hook_function))
      elif "hook_mlp_out" in hook_name:
        # the output of the MLP is written directly to the residual stream
        hooks.append((hook_name, write_to_resid_hook_function))
      elif "hook_mlp_in" in hook_name:
        # the input of the MLP is read directly from the residual stream
        hooks.append((hook_name, read_from_resid_hook_function))

    # Add hooks for the pre-residual stream and the post-residual stream of all layers, so that the residual stream has
    # the same size as the original model.
    # Also, TransformerLens does not have a hook for the input of the unembedding, so we need to specially need to
    # change the dimension of the last residual stream back to whatever it was before compression.
    for layer in range(self.cfg.n_layers):
      hooks.append((f"blocks.{layer}.hook_resid_pre", write_to_resid_hook_function))
      hooks.append((f"blocks.{layer}.hook_resid_post", read_from_resid_hook_function))

    return hooks

  def dump_compression_matrix(self, output_dir: str, filename: str):
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)

    compression_matrix = self.W_compress.weight.detach().cpu().numpy()
    np.save(os.path.join(output_dir, filename), compression_matrix)
