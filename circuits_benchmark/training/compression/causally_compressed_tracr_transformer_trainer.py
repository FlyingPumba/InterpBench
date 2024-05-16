import random
from typing import List, Dict

import torch as t
import wandb
from jaxtyping import Float
from torch import Tensor
from torch.nn import Parameter

from circuits_benchmark.benchmark.benchmark_case import BenchmarkCase
from circuits_benchmark.benchmark.case_dataset import CaseDataset
from circuits_benchmark.metrics.resampling_ablation_loss.resample_ablation_loss import \
  get_resample_ablation_loss_from_inputs
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

    if self.train_loss_level == "intervention":
      self.epochs_since_last_train_resample_ablation_loss = self.args.resample_ablation_loss_epochs_gap

  def compute_train_loss(self, batch: Dict[str, HookedTracrTransformerBatchInput]) -> Float[Tensor, ""]:
    # Run the input on both compressed and original model
    inputs = batch[CaseDataset.INPUT_FIELD]
    compressed_model_logits, compressed_model_cache = self.get_logits_and_cache_from_compressed_model(inputs)
    original_model_logits, original_model_cache = self.get_logits_and_cache_from_original_model(inputs)

    # Independently of the train loss level, we always compute the output loss since we want the compressed model to
    # have the same output as the original model.
    loss = self.get_output_loss(compressed_model_logits, original_model_logits)

    if self.train_loss_level == "layer":
      loss = loss + self.get_layer_level_loss(compressed_model_cache, original_model_cache)

    elif self.train_loss_level == "component":
      loss = loss + self.get_component_level_loss(compressed_model_cache, original_model_cache)

    elif self.train_loss_level == "intervention":
      if self.epochs_since_last_train_resample_ablation_loss >= self.args.resample_ablation_loss_epochs_gap:
        self.epochs_since_last_train_resample_ablation_loss = 0

        intervention_loss = self.get_intervention_level_loss()
        loss = loss + self.args.resample_ablation_loss_weight * intervention_loss

      self.epochs_since_last_train_resample_ablation_loss += 1

    else:
      raise NotImplementedError(f"Train loss level {self.train_loss_level} not implemented")

    if self.use_wandb:
      wandb.log({"train_loss": loss}, step=self.step)

    return loss

  def get_output_loss(self, compressed_model_logits, original_model_logits):
    if self.is_categorical:
      # Cross entropy loss
      loss = t.nn.functional.cross_entropy(compressed_model_logits.flatten(end_dim=-2),
                                           original_model_logits.flatten(end_dim=-2))
    else:
      # MSE loss
      loss = t.nn.functional.mse_loss(compressed_model_logits, original_model_logits)

    if self.use_wandb:
      wandb.log({"output_loss": loss}, step=self.step)

    return loss

  def get_layer_level_loss(self, compressed_model_cache, original_model_cache):
    loss = t.tensor(0.0, device=self.device)

    # Sum the L2 of output vectors for all layers in both compressed and original model
    for layer in range(self.n_layers):
      compressed_model_activations = compressed_model_cache["resid_post", layer]
      original_model_activations = original_model_cache["resid_post", layer]

      layer_loss = t.nn.functional.mse_loss(compressed_model_activations, original_model_activations)
      if self.use_wandb:
        wandb.log({f"layer_{str(layer)}_loss": layer_loss}, step=self.step)

      loss += layer_loss

    return loss

  def get_component_level_loss(self, compressed_model_cache, original_model_cache):
    loss = t.tensor(0.0, device=self.device)

    # Sum the L2 output vectors for all components (Attention heads and MLPS) in both compressed and original model
    for layer in range(self.n_layers):
      for component in ["attn", "mlp"]:
        hook_name = f"{component}_out"
        compressed_model_activations = compressed_model_cache[hook_name, layer]
        original_model_activations = original_model_cache[hook_name, layer]

        component_loss = t.nn.functional.mse_loss(original_model_activations, compressed_model_activations)
        if self.use_wandb:
          wandb.log({f"layer_{str(layer)}_{component}_loss": component_loss}, step=self.step)

    # Sum the L2 output vectors for the embeddings in both compressed and original model
    for component in ["embed", "pos_embed"]:
      hook_name = f"hook_{component}"
      compressed_model_activations = compressed_model_cache[hook_name]
      original_model_activations = original_model_cache[hook_name]

      component_loss = t.nn.functional.mse_loss(original_model_activations, compressed_model_activations)
      if self.use_wandb:
        wandb.log({f"{hook_name}_loss": component_loss}, step=self.step)

      loss += component_loss

    return loss

  def get_intervention_level_loss(self):
    resample_ablation_loss_args = {
      "clean_inputs": self.case.get_clean_data(count=self.args.resample_ablation_data_size,
                                               seed=random.randint(0, 1000000)),
      "corrupted_inputs": self.case.get_corrupted_data(count=self.args.resample_ablation_data_size,
                                                       seed=random.randint(0, 1000000)),
      "base_model": self.get_original_model(),
      "hypothesis_model": self.get_compressed_model(),
      "max_interventions": self.args.resample_ablation_max_interventions,
      "max_components": self.args.resample_ablation_max_components,
      "is_categorical": self.is_categorical,
      "batch_size": self.args.resample_ablation_batch_size,
    }

    activation_mapper = self.get_activation_mapper()
    if activation_mapper is not None:
      resample_ablation_loss_args["activation_mapper"] = activation_mapper

    resample_ablation_output = get_resample_ablation_loss_from_inputs(**resample_ablation_loss_args)

    if self.use_wandb:
      wandb.log({"train_resample_ablation_loss": resample_ablation_output.loss,
                 "train_resample_ablation_var_exp": resample_ablation_output.variance_explained}, step=self.step)

      for hook_name, loss in resample_ablation_output.max_loss_per_hook.items():
        wandb.log({f"train_{hook_name}_max_cp_loss": loss}, step=self.step)

      for hook_name, loss in resample_ablation_output.mean_loss_per_hook.items():
        wandb.log({f"train_{hook_name}_mean_cp_loss": loss}, step=self.step)

    return resample_ablation_output.loss

  def define_wandb_metrics(self):
    super().define_wandb_metrics()

    if self.train_loss_level == "layer":
      wandb.define_metric("output_loss", summary="min")
      for layer in range(self.n_layers):
        wandb.define_metric(f"layer_{str(layer)}_loss", summary="min")

    elif self.train_loss_level == "component":
      wandb.define_metric("output_loss", summary="min")
      for layer in range(self.n_layers):
        for component in ["attn", "mlp"]:
          wandb.define_metric(f"layer_{str(layer)}_{component}_loss", summary="min")
      for component in ["embed", "pos_embed"]:
        wandb.define_metric(f"hook_{component}_loss", summary="min")

    elif self.train_loss_level == "intervention":
      wandb.define_metric("train_resample_ablation_loss", summary="min")
      wandb.define_metric("train_resample_ablation_var_exp", summary="max")

    else:
      raise NotImplementedError(f"Train loss level {self.train_loss_level} not implemented")

  def get_wandb_config(self):
    cfg = super().get_wandb_config()
    cfg.update({
      "is_categorical": self.is_categorical,
      "n_layers": self.n_layers,
      "original_resid_size": self.get_original_model().cfg.d_model,
      "compressed_resid_size": self.get_compressed_model().cfg.d_model,
    })
    return cfg
