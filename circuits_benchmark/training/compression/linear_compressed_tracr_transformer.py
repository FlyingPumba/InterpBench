from __future__ import annotations

import os
import typing
from typing import List, Tuple, Callable, Literal, Iterator

import numpy as np
import torch as t
import wandb
from einops import einsum
from jaxtyping import Float
from torch import nn, Tensor
from torch.nn import Linear, Parameter, init
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint
from wandb.sdk.wandb_run import Run

from circuits_benchmark.transformers.hooked_tracr_transformer import HookedTracrTransformer

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
               *args, **kwargs):
    super().__init__(
      cfg=tl_model.cfg,
      tracr_input_encoder=tl_model.tracr_input_encoder,
      tracr_output_encoder=tl_model.tracr_output_encoder,
      residual_stream_labels=tl_model.residual_stream_labels,
      *args, **kwargs)
    self.original_residual_stream_size = tl_model.cfg.d_model
    self.residual_stream_compression_size = residual_stream_compression_size

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

  def folded_named_parameters(
            self,
            prefix: str = '',
            recurse: bool = True,
            remove_duplicate: bool = True
    ) -> Iterator[Tuple[str, Parameter]]:
    """Returns an iterator over module parameters, yielding both the name of the parameter as well as the parameter.
    This is a modified version of the original named_parameters method, which folds the compression matrix into the
    named parameters."""
    named_params = list(super().named_parameters(prefix, recurse, remove_duplicate))

    # if we don't have self.W_compress yet (e.g. when we are in the constructor), we return the original named params
    if not hasattr(self, "W_compress"):
      for name, param in named_params:
        yield name, param
      return

    compression_matrix = self.W_compress.weight

    for name, param in named_params:
      if name == "embed.W_E" or name == "pos_embed.W_pos": # shape: [vocab_size, d_model] and [max_seq_len, d_model]
        param = Parameter(param.clone())
        param.data = param.data @ compression_matrix
        yield name, param

      elif "W_Q" in name or "W_K" in name or "W_V" in name: # shape: [n_heads, d_model, d_head]
        param = Parameter(param.clone())
        param.data = einsum(param.data, compression_matrix,
                            "n_heads d_model d_head, d_model d_compressed_model -> n_heads d_compressed_model d_head")
        yield name, param

      elif "W_O" in name: # shape: [n_heads, d_head, d_model]
        param = Parameter(param.clone())
        param.data = einsum(param.data, compression_matrix,
                            "n_heads d_head d_model, d_model d_compressed_model -> n_heads d_head d_compressed_model")
        yield name, param

      elif "b_Q" in name or "b_K" in name or "b_V" in name: # shape: [n_heads, d_head]
        # no changes
        yield name, param

      elif "b_O" in name: # shape: [n_heads, d_model]
        param = Parameter(param.clone())
        param.data = einsum(param.data.unsqueeze(dim=0), compression_matrix,
                            "n_heads d_model, d_model d_compressed_model -> n_heads d_compressed_model").squeeze(dim=0)
        yield name, param

      elif "W_in" in name: # shape: [d_model, d_mlp]
        param = Parameter(param.clone())
        param.data = einsum(param.data, compression_matrix,
                            "d_model d_mlp, d_model d_compressed_model -> d_compressed_model d_mlp")
        yield name, param

      elif "W_out" in name: # shape: [d_mlp, d_model]
        param = Parameter(param.clone())
        param.data = einsum(param.data, compression_matrix,
                            "d_mlp d_model, d_model d_compressed_model -> d_mlp d_compressed_model")
        yield name, param

      elif "b_in" in name: # shape: [d_mlp]
        # no changes
        yield name, param

      elif "b_out" in name: # shape: [d_model]
        param = Parameter(param.clone())
        param.data = einsum(param.data.unsqueeze(dim=0), compression_matrix,
                            "_ d_model, d_model d_compressed_model -> _ d_compressed_model").squeeze(dim=0)
        yield name, param

      elif "unembed.W_U" in name: # shape: [d_model, d_out]
        param = Parameter(param.clone())
        param.data = einsum(param.data, compression_matrix,
                            "d_model d_out, d_model d_compressed_model -> d_compressed_model d_out")
        yield name, param

      elif "b_U" in name: # shape: [d_out]
        # no changes
        yield name, param

      elif "W_compress" in name:
        # ignore the compression matrix
        continue

      else:
        raise ValueError(f"Unknown parameter name: {name}")

  def folded_parameters(self, recurse: bool = True) -> Iterator[Parameter]:
    for name, param in self.named_parameters(recurse=recurse):
      yield param

  def get_folded_model(self) -> HookedTracrTransformer:
    tl_model = HookedTracrTransformer.from_hooked_tracr_transformer(
      self,
      overwrite_cfg_dict={"d_model": self.residual_stream_compression_size},
      init_params_fn=lambda x: init.kaiming_uniform_(x) if len(x.shape) > 1 else init.normal_(x, std=0.02),
    )

    for name, param in self.folded_named_parameters():
      matching_param = next(tl_param for tl_name, tl_param in tl_model.named_parameters() if tl_name == name)
      # assert they have the same shape
      assert param.shape == matching_param.shape
      matching_param.data = param.data

    return tl_model

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
