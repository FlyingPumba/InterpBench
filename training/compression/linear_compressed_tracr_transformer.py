from __future__ import annotations

import os
import typing
from typing import List, Tuple, Callable, Literal

import numpy as np
import torch as t
import wandb
from jaxtyping import Float
from torch import nn, Tensor
from torch.nn import Linear
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint
from wandb.sdk.wandb_run import Run

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

    self.W_compress: Linear = nn.Linear(self.residual_stream_compression_size,
                                        self.original_residual_stream_size,
                                        device=self.device,
                                        bias=False)

    if linear_compression_initialization == "orthogonal":
      # The (semi) orthogonal matrix is useful for our setup because the transpose is exactly the inverse, and we will
      # use the same matrix for reading and writing from/to the residual stream.
      nn.init.orthogonal_(self.W_compress.weight)

    assert self.W_compress.weight.shape == (self.original_residual_stream_size, self.residual_stream_compression_size)

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
    compression_matrix = self.W_compress.weight

    def compress_resid_hook_fn(
        residual_stream: Float[Tensor, "batch seq_len d_model"],
        hook: HookPoint
    ) -> Float[Tensor, "batch seq_len d_model_compressed"]:
      return residual_stream @ compression_matrix

    def decompress_resid_hook_fn(
        residual_stream: Float[Tensor, "batch seq_len n_head d_model_compressed"],
        hook: HookPoint
    ) -> Float[Tensor, "batch seq_len n_head d_model"]:
      return residual_stream @ compression_matrix.T

    # Initial hooks for the input and positional embeddings, so that the first residual stream has compressed size.
    hooks.append((f"hook_embed", compress_resid_hook_fn))
    hooks.append((f"hook_pos_embed", compress_resid_hook_fn))

    # Add hooks for Attention heads and MLPs.
    # Each input hook decompresses the residual stream, and each output hook compresses back the residual stream.
    for hook_name in self.hook_dict.keys():
      if "hook_k_input" in hook_name or "hook_q_input" in hook_name or "hook_v_input" in hook_name:
        # Attention head matrices read directly from the residual stream
        hooks.append((hook_name, decompress_resid_hook_fn))
      elif "hook_attn_out" in hook_name:
        # the output of attention heads is written directly to the residual stream
        hooks.append((hook_name, compress_resid_hook_fn))
      elif "hook_mlp_out" in hook_name:
        # the output of the MLP is written directly to the residual stream
        hooks.append((hook_name, compress_resid_hook_fn))
      elif "hook_mlp_in" in hook_name:
        # the input of the MLP is read directly from the residual stream
        hooks.append((hook_name, decompress_resid_hook_fn))

    # TransformerLens does not have a hook for the input of the unembedding, so we need to decompress the residual
    # stream produced by the last layer.
    hooks.append((f"blocks.{self.cfg.n_layers - 1}.hook_resid_post", decompress_resid_hook_fn))

    return hooks

  def save(self, output_dir: str, prefix: str, wandb_run: Run | None = None):
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)

    # save compression matrix by its own
    compression_matrix = self.W_compress.weight.detach().cpu().numpy()
    compression_matrix_path = os.path.join(output_dir, f"{prefix}-linear-compression-matrix.npy")
    np.save(compression_matrix_path, compression_matrix)

    # save the weights of the model using state_dict
    weights_path = os.path.join(output_dir, f"{prefix}-linear-compression-weights.pt")
    t.save(self.state_dict(), weights_path)

    # save the entire model
    # The following is commented out due to a pickle error: "Can't pickle local object 'HookPoint.add_hook.<locals>.full_hook'"
    # model_path = os.path.join(output_dir, f"{prefix}-linearly-compressed-tracr-transformer.pt")
    # t.save(self, model_path)

    if wandb_run is not None:
      # save the files as artifacts to wandb
      artifact = wandb.Artifact(f"{prefix}-linearly-compressed-tracr-transformer", type="model")
      artifact.add_file(weights_path)
      artifact.add_file(compression_matrix_path)
      # artifact.add_file(model_path)
      wandb_run.log_artifact(artifact)
