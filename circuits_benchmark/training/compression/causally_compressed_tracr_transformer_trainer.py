from typing import List, Dict

import torch as t
import wandb
from jaxtyping import Float
from torch import Tensor
from torch.nn import Parameter

from circuits_benchmark.benchmark.benchmark_case import BenchmarkCase
from circuits_benchmark.benchmark.case_dataset import CaseDataset
from circuits_benchmark.training.compression.compressed_tracr_transformer_trainer import \
  CompressedTracrTransformerTrainer
from circuits_benchmark.training.compression.compression_train_loss_level import CompressionTrainLossLevel
from circuits_benchmark.training.training_args import TrainingArgs
from circuits_benchmark.transformers.hooked_tracr_transformer import HookedTracrTransformerBatchInput


class CausallyCompressedTracrTransformerTrainer(CompressedTracrTransformerTrainer):

  def __init__(self,
               case: BenchmarkCase,
               parameters: List[Parameter],
               training_args: TrainingArgs,
               is_categorical: bool,
               n_layers: int,
               train_loss_level: CompressionTrainLossLevel = "layer",
               output_dir: str | None = None):
    super().__init__(case, parameters, training_args, is_categorical, n_layers, output_dir=output_dir)
    self.train_loss_level = train_loss_level

  def compute_train_loss(self, batch: Dict[str, HookedTracrTransformerBatchInput]) -> Float[Tensor, ""]:
    # Run the input on both compressed and original model
    inputs = batch[CaseDataset.INPUT_FIELD]
    compressed_model_logits, compressed_model_cache = self.get_logits_and_cache_from_compressed_model(inputs)
    original_model_logits, original_model_cache = self.get_logits_and_cache_from_original_model(inputs)

    if self.is_categorical:
      # Cross entropy loss
      loss = t.nn.functional.cross_entropy(compressed_model_logits.flatten(end_dim=-2),
                                           original_model_logits.flatten(end_dim=-2))
    else:
      # MSE loss
      loss = t.nn.functional.mse_loss(compressed_model_logits, original_model_logits)

    if self.use_wandb:
      wandb.log({"output_loss": loss}, step=self.step)

    if self.train_loss_level == "layer":
      # Sum the L2 of output vectors for all layers in both compressed and original model
      for layer in range(self.n_layers):
        compressed_model_output = compressed_model_cache["resid_post", layer]
        original_model_output = original_model_cache["resid_post", layer]

        layer_loss = t.nn.functional.mse_loss(compressed_model_output, original_model_output)
        if self.use_wandb:
          wandb.log({f"layer_{str(layer)}_loss": layer_loss}, step=self.step)

        loss += layer_loss

    elif self.train_loss_level == "component":
      # Sum the L2 output vectors for all components (Attention heads and MLPS) in both compressed and original model
      for layer in range(self.n_layers):
        for component in ["attn", "mlp"]:
          hook_name = f"{component}_out"
          compressed_model_output = compressed_model_cache[hook_name, layer]
          original_model_output = original_model_cache[hook_name, layer]

          component_loss = t.nn.functional.mse_loss(original_model_output, compressed_model_output)
          if self.use_wandb:
            wandb.log({f"layer_{str(layer)}_{component}_loss": component_loss}, step=self.step)

          loss += component_loss

      # Sum the L2 output vectors for the embeddings in both compressed and original model
      for component in ["embed", "pos_embed"]:
        hook_name = f"hook_{component}"
        compressed_model_output = compressed_model_cache[hook_name]
        original_model_output = original_model_cache[hook_name]

        component_loss = t.nn.functional.mse_loss(original_model_output, compressed_model_output)
        if self.use_wandb:
          wandb.log({f"{hook_name}_loss": component_loss}, step=self.step)

        loss += component_loss

    else:
      raise NotImplementedError(f"Train loss level {self.train_loss_level} not implemented")

    if self.use_wandb:
      wandb.log({"train_loss": loss}, step=self.step)

    return loss

  def get_wandb_config(self):
    cfg = super().get_wandb_config()
    cfg.update({
      "is_categorical": self.is_categorical,
      "n_layers": self.n_layers,
      "original_resid_size": self.get_original_model().cfg.d_model,
      "compressed_resid_size": self.get_compressed_model().cfg.d_model,
    })
    return cfg
