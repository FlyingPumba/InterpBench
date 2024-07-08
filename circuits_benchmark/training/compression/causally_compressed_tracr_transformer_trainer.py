from typing import List

import torch as t
import wandb
from jaxtyping import Float, Int
from torch import Tensor
from torch.nn import Parameter

from circuits_benchmark.benchmark.benchmark_case import BenchmarkCase
from circuits_benchmark.metrics.resampling_ablation_loss.resample_ablation_loss import \
  get_resample_ablation_loss
from circuits_benchmark.training.compression.compressed_tracr_transformer_trainer import \
  CompressedTracrTransformerTrainer
from circuits_benchmark.training.training_args import TrainingArgs
from circuits_benchmark.utils.iit.iit_dataset_batch import IITDatasetBatch


class CausallyCompressedTracrTransformerTrainer(CompressedTracrTransformerTrainer):

  def __init__(self,
               case: BenchmarkCase,
               parameters: List[Parameter],
               training_args: TrainingArgs,
               is_categorical: bool,
               n_layers: int,
               output_dir: str | None = None):
    self.last_resample_ablation_loss = None
    self.interventions_per_node = {}

    super().__init__(case, parameters, training_args, is_categorical, n_layers, output_dir=output_dir)

    self.epochs_since_last_train_resample_ablation_loss = self.args.resample_ablation_loss_epochs_gap

  def compute_train_loss(self, batch: IITDatasetBatch) -> Float[Tensor, ""]:
    clean_data, corrupted_data = batch
    clean_inputs = clean_data[0]

    # We always compute the output loss since we want the compressed model to
    # have the same output as the original model.
    compressed_model_logits = self.get_compressed_model()(clean_inputs)
    original_model_logits = self.get_original_model()(clean_inputs)
    loss = self.get_output_loss(compressed_model_logits, original_model_logits)

    if self.epochs_since_last_train_resample_ablation_loss >= self.args.resample_ablation_loss_epochs_gap:
      self.epochs_since_last_train_resample_ablation_loss = 0

      intervention_loss = self.get_intervention_level_loss(batch)
      loss = loss + self.args.resample_ablation_loss_weight * intervention_loss

    self.epochs_since_last_train_resample_ablation_loss += 1

    if self.use_wandb:
      wandb.log({"train_loss": loss}, step=self.step)

    return loss

  def get_output_loss(self, compressed_model_logits, original_model_logits):
    # The output has unspecified behavior for the BOS token, so we discard it on the loss calculation.
    compressed_model_logits = compressed_model_logits[:, 1:]
    original_model_logits = original_model_logits[:, 1:]

    if self.is_categorical:
      # Cross entropy loss
      flattened_logits: Float[Tensor, "batch*pos, vocab_out"] = compressed_model_logits.flatten(end_dim=-2)
      flattened_expected_labels: Int[Tensor, "batch*pos"] = original_model_logits.argmax(dim=-1).flatten()
      loss = t.nn.functional.cross_entropy(flattened_logits, flattened_expected_labels)
    else:
      # MSE loss
      loss = t.nn.functional.mse_loss(compressed_model_logits, original_model_logits)

    if self.use_wandb:
      wandb.log({"output_loss": loss}, step=self.step)
      wandb.log({"base_model_logits_std": original_model_logits.std()}, step=self.step)
      wandb.log({"compressed_model_logits_std": compressed_model_logits.std()}, step=self.step)

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

  def get_intervention_level_loss(self, batch: IITDatasetBatch):
    resample_ablation_loss_args = {
      "data": batch,
      "base_model": self.get_original_model(),
      "hypothesis_model": self.get_compressed_model(),
      "max_interventions": self.args.resample_ablation_max_interventions,
      "max_components": self.args.resample_ablation_max_components,
      "is_categorical": self.is_categorical,
    }

    activation_mapper = self.get_activation_mapper()
    if activation_mapper is not None:
      resample_ablation_loss_args["activation_mapper"] = activation_mapper

    if self.effect_diffs_by_node is not None:
      resample_ablation_loss_args["effect_diffs_by_node"] = self.effect_diffs_by_node

    resample_ablation_output = get_resample_ablation_loss(**resample_ablation_loss_args)

    if self.use_wandb:
      wandb.log({"train_resample_ablation_loss": resample_ablation_output.loss,
                 "train_resample_ablation_var_exp": resample_ablation_output.variance_explained}, step=self.step)

      for hook_name, loss in resample_ablation_output.max_loss_per_node.items():
        wandb.log({f"train_{hook_name}_max_cp_loss": loss}, step=self.step)

      for hook_name, loss in resample_ablation_output.mean_loss_per_node.items():
        wandb.log({f"train_{hook_name}_mean_cp_loss": loss}, step=self.step)

      for node_name, interventions_count in resample_ablation_output.interventions_per_node.items():
        if node_name in self.interventions_per_node:
          self.interventions_per_node[node_name] += interventions_count
        else:
          self.interventions_per_node[node_name] = interventions_count

        wandb.log({f"{node_name}_interventions_count": self.interventions_per_node[node_name]}, step=self.step)

    self.last_resample_ablation_loss = resample_ablation_output.loss.item()

    return resample_ablation_output.loss

  def define_wandb_metrics(self):
    super().define_wandb_metrics()

    wandb.define_metric("train_resample_ablation_loss", summary="min")
    wandb.define_metric("train_resample_ablation_var_exp", summary="max")

  def get_wandb_config(self):
    cfg = super().get_wandb_config()
    cfg.update({
      "is_categorical": self.is_categorical,
      "n_layers": self.n_layers,
      "original_resid_size": self.get_original_model().cfg.d_model,
      "compressed_resid_size": self.get_compressed_model().cfg.d_model,
    })
    return cfg
