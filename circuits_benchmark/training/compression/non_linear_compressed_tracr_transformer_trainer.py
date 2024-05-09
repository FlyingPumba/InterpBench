import dataclasses
import os
from typing import Dict

import torch as t
import transformer_lens.utils as utils
import wandb
from jaxtyping import Float
from torch import Tensor
from transformer_lens import ActivationCache, HookedTransformer

from circuits_benchmark.benchmark.benchmark_case import BenchmarkCase
from circuits_benchmark.training.compression.activation_mapper.activation_mapper import ActivationMapper
from circuits_benchmark.training.compression.activation_mapper.autoencoder_mapper import AutoEncoderMapper
from circuits_benchmark.training.compression.activation_mapper.multi_hook_activation_mapper import MultiHookActivationMapper
from circuits_benchmark.training.compression.autencoder import AutoEncoder
from circuits_benchmark.training.compression.autoencoder_trainer import AutoEncoderTrainer
from circuits_benchmark.training.compression.causally_compressed_tracr_transformer_trainer import \
  CausallyCompressedTracrTransformerTrainer
from circuits_benchmark.training.compression.compression_train_loss_level import CompressionTrainLossLevel
from circuits_benchmark.training.training_args import TrainingArgs
from circuits_benchmark.transformers.hooked_tracr_transformer import HookedTracrTransformer, \
  HookedTracrTransformerBatchInput


class NonLinearCompressedTracrTransformerTrainer(CausallyCompressedTracrTransformerTrainer):
  def __init__(self, case: BenchmarkCase,
               old_tl_model: HookedTracrTransformer,
               new_tl_model: HookedTracrTransformer,
               autoencoders_dict: Dict[str, AutoEncoder],
               args: TrainingArgs,
               train_loss_level: CompressionTrainLossLevel = "layer",
               output_dir: str | None = None,
               freeze_ae_weights: bool = False,
               ae_training_epochs_gap: int = 10,
               ae_desired_test_mse: float = 1e-3,
               ae_max_training_epochs: int = 15,
               ae_training_args: TrainingArgs = None):
    self.old_tl_model: HookedTracrTransformer = old_tl_model
    self.new_tl_model: HookedTracrTransformer = new_tl_model
    self.autoencoders_dict: Dict[str, AutoEncoder] = autoencoders_dict
    self.autoencoder_trainers_dict: Dict[str, AutoEncoderTrainer] = {}
    self.device = old_tl_model.device

    parameters = list(new_tl_model.parameters())
    self.freeze_ae_weights = freeze_ae_weights
    if self.freeze_ae_weights:
      self.autoencoder_trainers_dict = None
      for ae in self.autoencoders_dict.values():
        ae.freeze_all_weights()
    else:
      # We will train the autoencoder every fixed number of epochs for the transformer training.
      for ae in self.autoencoders_dict.values():
        parameters += list(ae.parameters())
      self.ae_training_epochs_gap = ae_training_epochs_gap
      self.ae_max_training_epochs = ae_max_training_epochs
      self.ae_desired_test_mse = ae_desired_test_mse
      self.epochs_since_last_ae_training = 0

      # make a copy of the training args for non-linear compression if AE specific training args were not provided
      self.ae_training_args = ae_training_args
      if self.ae_training_args is None:
        self.ae_training_args = dataclasses.replace(args, wandb_project=None, wandb_name=None)

      for ae_key, ae in self.autoencoders_dict.items():
        self.autoencoder_trainers_dict[ae_key] = AutoEncoderTrainer(case, ae,
                                                                 self.old_tl_model,
                                                                 self.ae_training_args,
                                                                 train_loss_level=train_loss_level,
                                                                 output_dir=output_dir)

    super().__init__(case,
                     parameters,
                     args,
                     old_tl_model.is_categorical(),
                     new_tl_model.cfg.n_layers,
                     train_loss_level=train_loss_level,
                     output_dir=output_dir)

  def training_epoch(self):
    if not self.freeze_ae_weights and self.epochs_since_last_ae_training >= self.ae_training_epochs_gap:
      self.epochs_since_last_ae_training = 0
      self.train_autoencoders()

    super().training_epoch()

    if not self.freeze_ae_weights:
      self.epochs_since_last_ae_training += 1

  def train_autoencoders(self):
    avg_ae_train_loss = None

    for ae_key, ae_trainer in self.autoencoder_trainers_dict.items():
        ae_trainer.compute_test_metrics()
        ae_training_epoch = 0
        while (ae_trainer.test_metrics["test_mse"] > self.ae_desired_test_mse and
               ae_training_epoch < self.ae_max_training_epochs):
          ae_train_losses = []
          for i, batch in enumerate(ae_trainer.train_loader):
            ae_train_loss = ae_trainer.training_step(batch)
            ae_train_losses.append(ae_train_loss)

          avg_ae_train_loss = t.mean(t.stack(ae_train_losses))

          ae_trainer.compute_test_metrics()
          ae_training_epoch += 1

        if self.use_wandb and avg_ae_train_loss is not None:
          # We performed training for the AutoEncoder. Log average train loss and test metrics
          wandb.log({f"ae_{ae_key}_train_loss": avg_ae_train_loss}, step=self.step)
          wandb.log({f"ae_{ae_key}_{k}": v for k, v in ae_trainer.test_metrics.items()}, step=self.step)

  def get_decoded_outputs_from_compressed_model(self, inputs: HookedTracrTransformerBatchInput) -> Tensor:
    return self.new_tl_model(inputs, return_type="decoded")

  def get_logits_and_cache_from_compressed_model(
      self,
      inputs: HookedTracrTransformerBatchInput
  ) -> (Float[Tensor, "batch seq_len d_vocab"], ActivationCache):
    compressed_model_logits, compressed_model_cache = self.new_tl_model.run_with_cache(inputs)

    if self.train_loss_level == "layer":
      # Decompress the residual streams of all layers except the last one, which we have already decompressed for using
      # the unembedding since TransformerLens does not have a hook for that.
      for layer in range(self.n_layers):
        cache_key = utils.get_act_name("resid_post", layer)
        compressed_model_cache.cache_dict[cache_key] = self.get_activation_mapper().decompress(
          compressed_model_cache.cache_dict[cache_key])

    elif self.train_loss_level == "component":
      # Decompress the residual streams outputted by the attention and mlp components of all layers
      for layer in range(self.n_layers):
        for component in ["attn", "mlp"]:
          hook_name = f"{component}_out"
          cache_key = utils.get_act_name(hook_name, layer)
          compressed_model_cache.cache_dict[cache_key] = self.get_activation_mapper().decompress(
            compressed_model_cache.cache_dict[cache_key])

      for component in ["embed", "pos_embed"]:
        hook_name = f"hook_{component}"
        compressed_model_cache.cache_dict[hook_name] = self.get_activation_mapper().decompress(
          compressed_model_cache.cache_dict[hook_name])

    elif self.train_loss_level == "intervention":
      return compressed_model_logits, compressed_model_cache

    else:
      raise ValueError(f"Invalid train loss level: {self.train_loss_level}")

    return compressed_model_logits, compressed_model_cache

  def get_logits_and_cache_from_original_model(
      self,
      inputs: HookedTracrTransformerBatchInput
  ) -> (Float[Tensor, "batch seq_len d_vocab"], ActivationCache):
    return self.old_tl_model.run_with_cache(inputs)

  def get_original_model(self) -> HookedTransformer:
    return self.old_tl_model

  def get_compressed_model(self) -> HookedTransformer:
    return self.new_tl_model

  def get_activation_mapper(self) -> MultiHookActivationMapper | ActivationMapper | None:
    mappers_dict = {k: AutoEncoderMapper(v) for k, v in self.autoencoders_dict.items()}
    return MultiHookActivationMapper(mappers_dict)

  def build_wandb_name(self):
    if len(self.autoencoders_dict) > 1:
      return f"case-{self.case.get_index()}-non-linear-multi-aes"
    else:
      return f"case-{self.case.get_index()}-non-linear-resid-{list(self.autoencoders_dict.values())[0].compression_size}"

  def get_wandb_tags(self):
    tags = super().get_wandb_tags()
    tags.append("non-linear-compression-trainer")
    return tags

  def get_wandb_config(self):
    cfg = super().get_wandb_config()
    cfg.update({
      "freeze_ae_weights": self.freeze_ae_weights,
    })
    for ae_key, ae in self.autoencoders_dict.items():
      cfg.update({
        f"ae_{ae_key}_input_size": ae.input_size,
        f"ae_{ae_key}_compression_size": ae.compression_size,
        f"ae_{ae_key}_layers": ae.n_layers,
        f"ae_{ae_key}_first_hidden_layer_shape": ae.first_hidden_layer_shape,
        f"ae_{ae_key}_use_bias": ae.use_bias,
      })

    if not self.freeze_ae_weights:
      cfg.update({
        "ae_training_epochs_gap": self.ae_training_epochs_gap,
        "ae_max_training_epochs": self.ae_max_training_epochs,
        "ae_desired_test_mse": self.ae_desired_test_mse,
      })
      cfg.update({f"ae_training_args_{k}": v for k, v in dataclasses.asdict(self.ae_training_args).items()})

    return cfg

  def save_artifacts(self):
    if not os.path.exists(self.output_dir):
      os.makedirs(self.output_dir)

    if len(self.autoencoders_dict) > 1:
      prefix = f"case-{self.case.get_index()}-multi-aes"
    else:
      prefix = f"case-{self.case.get_index()}-resid-{list(self.autoencoders_dict.values())[0].compression_size}"


    # save the weights of the model using state_dict
    weights_path = os.path.join(self.output_dir, f"{prefix}-non-linear-compression-weights.pt")
    t.save(self.get_compressed_model().state_dict(), weights_path)

    # save the entire model
    model_path = os.path.join(self.output_dir, f"{prefix}-non-linearly-compressed-tracr-transformer.pt")
    t.save(self.get_compressed_model(), model_path)

    if self.wandb_run is not None:
      # save the files as artifacts to wandb
      artifact = wandb.Artifact(f"{prefix}-non-linearly-compressed-tracr-transformer", type="model")
      artifact.add_file(weights_path)
      artifact.add_file(model_path)
      self.wandb_run.log_artifact(artifact)

    if not self.freeze_ae_weights:
      # The autoencoders have changed during the non-linear compression training, so we will save it.
      for ae_key, ae in self.autoencoders_dict.items():
        prefix = f"case-{self.case.get_index()}-ae-{ae_key}-size-{ae.compression_size}-final"
        ae.save(self.output_dir, prefix, self.wandb_run)