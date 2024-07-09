import dataclasses
import os
from typing import Dict

import torch as t
import wandb
from iit.model_pairs.ll_model import LLModel
from jaxtyping import Float
from torch import Tensor
from transformer_lens import ActivationCache

from circuits_benchmark.benchmark.benchmark_case import BenchmarkCase
from circuits_benchmark.training.compression.activation_mapper.activation_mapper import ActivationMapper
from circuits_benchmark.training.compression.activation_mapper.autoencoder_mapper import AutoEncoderMapper
from circuits_benchmark.training.compression.activation_mapper.multi_hook_activation_mapper import \
  MultiHookActivationMapper
from circuits_benchmark.training.compression.autencoder import AutoEncoder
from circuits_benchmark.training.compression.autoencoder_trainer import AutoEncoderTrainer
from circuits_benchmark.training.compression.causally_compressed_tracr_transformer_trainer import \
  CausallyCompressedTracrTransformerTrainer
from circuits_benchmark.training.training_args import TrainingArgs
from circuits_benchmark.utils.iit.iit_dataset_batch import IITDatasetBatch


class NonLinearCompressedTracrTransformerTrainer(CausallyCompressedTracrTransformerTrainer):
  def __init__(self, case: BenchmarkCase,
               old_tl_model: LLModel,
               new_tl_model: LLModel,
               autoencoders_dict: Dict[str, AutoEncoder],
               args: TrainingArgs,
               output_dir: str | None = None,
               ae_desired_test_mse: float = 1e-3,
               ae_training_args: TrainingArgs = None,
               ae_train_loss_weight: float = 10):
    self.old_tl_model: LLModel = old_tl_model
    self.new_tl_model: LLModel = new_tl_model
    self.autoencoders_dict: Dict[str, AutoEncoder] = autoencoders_dict
    self.autoencoder_trainers_dict: Dict[str, AutoEncoderTrainer] = {}
    self.device = old_tl_model.device

    # freeze old_tl_model parameters
    for param in self.old_tl_model.parameters():
      param.requires_grad = False

    parameters = list(new_tl_model.parameters())
    for ae in self.autoencoders_dict.values():
      parameters += list(ae.parameters())

    # define this before calling the super constructor, since its needed for the wandb config
    self.ae_desired_test_mse = ae_desired_test_mse
    self.ae_train_loss_weight = ae_train_loss_weight
    self.ae_training_args = ae_training_args if ae_training_args is not None else args
    print(f"AutoEncoder training args for non-linear compression: {self.ae_training_args}")
    print(f"AutoEncoder desired test MSE for non-linear compression: {self.ae_desired_test_mse}")
    print(f"AutoEncoder training loss weight for non-linear compression: {self.ae_train_loss_weight}")

    super().__init__(case,
                     parameters,
                     args,
                     old_tl_model.is_categorical(),
                     new_tl_model.cfg.n_layers,
                     output_dir=output_dir)

    # make a first pass of AE training before starting the transformer training
    self.init_autoencoders()

  def init_autoencoders(self):
    """Perform initial training for the AutoEncoders."""
    print(" >>> Training the autoencoders before starting the transformer training.")

    # Collect activations for the autoencoders. We need to do this in batches to avoid memory issues.
    activations_cache_dict = {}
    with t.no_grad():
      for i, batch in enumerate(self.train_loader):
        _, batch_activations_cache_dict = self.old_tl_model.run_with_cache(batch)
        for key, value in batch_activations_cache_dict.items():
          if key not in activations_cache_dict:
            activations_cache_dict[key] = value
          else:
            activations_cache_dict[key] = t.cat([activations_cache_dict[key], value], dim=0)

    activations_cache = ActivationCache(activations_cache_dict, self.old_tl_model)

    for ae_key, ae in self.autoencoders_dict.items():
      ae_trainer = AutoEncoderTrainer(
        self.case,
        ae,
        self.old_tl_model,
        self.ae_training_args,
        activations_cache,
        hook_name_filter_for_input_activations=ae_key,
        output_dir=self.output_dir
      )
      self.autoencoder_trainers_dict[ae_key] = ae_trainer

    avg_ae_train_loss = None

    for ae_key, ae_trainer in self.autoencoder_trainers_dict.items():
      ae_trainer.compute_test_metrics()
      ae_training_epoch = 0
      while (ae_trainer.test_metrics["test_mse"] > self.ae_desired_test_mse and
             ae_training_epoch < self.ae_training_args.epochs):
        ae_train_losses = []
        for i, batch in enumerate(ae_trainer.train_loader):
          ae_train_loss = ae_trainer.training_step(batch)
          ae_train_losses.append(ae_train_loss)

        avg_ae_train_loss = t.mean(t.stack(ae_train_losses))

        ae_trainer.compute_test_metrics()
        ae_training_epoch += 1

      print(
        f"AutoEncoder {ae_key} trained for {ae_training_epoch} epochs, and achieved train loss of {avg_ae_train_loss}.")
      print(f"AutoEncoder {ae_key} test metrics: {ae_trainer.test_metrics}")

      if self.use_wandb and avg_ae_train_loss is not None:
        # We performed training for the AutoEncoder. Log average train loss and test metrics
        wandb.log({f"ae_{ae_key}_train_loss": avg_ae_train_loss}, step=self.step)
        wandb.log({f"ae_{ae_key}_{k}": v for k, v in ae_trainer.test_metrics.items()}, step=self.step)

  def compute_train_loss(self, batch: IITDatasetBatch) -> Float[Tensor, ""]:
    """Computes the training loss and adds the AEs reconstruction loss to it."""
    train_loss = super().compute_train_loss(batch)

    for ae_key, ae_trainer in self.autoencoder_trainers_dict.items():
      ae_train_losses = []
      for i, batch in enumerate(ae_trainer.train_loader):
        ae_train_loss = ae_trainer.compute_train_loss(batch)
        ae_train_losses.append(ae_train_loss)

      avg_ae_train_loss = t.mean(t.stack(ae_train_losses))
      train_loss += self.ae_train_loss_weight * avg_ae_train_loss

      if self.use_wandb and avg_ae_train_loss is not None:
        # We performed training for the AutoEncoder. Log average train loss and test metrics
        wandb.log({f"ae_{ae_key}_train_loss": avg_ae_train_loss}, step=self.step)

    return train_loss

  def compute_test_metrics(self):
    super().compute_test_metrics()

    # add AEs test metrics
    for ae_key, ae_trainer in self.autoencoder_trainers_dict.items():
      ae_trainer.compute_test_metrics()
      ae_specific_test_metrics = {f"ae_{ae_key}_{k}": v for k, v in ae_trainer.test_metrics.items()}
      self.test_metrics.update(ae_specific_test_metrics)

      if self.use_wandb:
        wandb.log(ae_specific_test_metrics, step=self.step)

    # Calculate average AE test MSE
    avg_ae_test_mse = 0
    for ae_key, ae_trainer in self.autoencoder_trainers_dict.items():
      avg_ae_test_mse += ae_trainer.test_metrics["test_mse"]
    avg_ae_test_mse /= len(self.autoencoder_trainers_dict)
    self.test_metrics["avg_ae_test_mse"] = avg_ae_test_mse
    if self.use_wandb:
      wandb.log({"avg_ae_test_mse": avg_ae_test_mse}, step=self.step)

    # log norm of all params individually
    if self.use_wandb:
      for name, param in self.new_tl_model.named_parameters():
        wandb.log({f"param_{name}_norm": param.norm()}, step=self.step)

  def get_original_model(self) -> LLModel:
    return self.old_tl_model

  def get_compressed_model(self) -> LLModel:
    return self.new_tl_model

  def get_activation_mapper(self) -> MultiHookActivationMapper | ActivationMapper | None:
    mappers_dict = {k: AutoEncoderMapper(v) for k, v in self.autoencoders_dict.items()}
    return MultiHookActivationMapper(mappers_dict)

  def build_test_metrics_string(self):
    return (f", iia: {self.test_metrics.get('iia', 0):.3f}, "
            f"siia: {self.test_metrics.get('siia', 0):.3f}, "
            f"avg_ae_test_mse: {self.test_metrics.get('avg_ae_test_mse', 0):.3f}")

  def build_wandb_name(self):
    if len(self.autoencoders_dict) > 1:
      return f"case-{self.case.get_name()}-non-linear-multi-aes"
    else:
      return f"case-{self.case.get_name()}-non-linear-resid-{list(self.autoencoders_dict.values())[0].compression_size}"

  def get_wandb_tags(self):
    tags = super().get_wandb_tags()
    tags.append("non-linear-compression-trainer")
    return tags

  def get_wandb_config(self):
    cfg = super().get_wandb_config()
    for ae_key, ae in self.autoencoders_dict.items():
      cfg.update({
        f"ae_{ae_key}_input_size": ae.input_size,
        f"ae_{ae_key}_compression_size": ae.compression_size,
        f"ae_{ae_key}_layers": ae.n_layers,
        f"ae_{ae_key}_first_hidden_layer_shape": ae.first_hidden_layer_shape,
        f"ae_{ae_key}_use_bias": ae.use_bias,
      })

    cfg.update({
      "ae_desired_test_mse": self.ae_desired_test_mse,
      "ae_train_loss_weight": self.ae_train_loss_weight,
    })
    cfg.update({f"ae_training_args_{k}": v for k, v in dataclasses.asdict(self.ae_training_args).items()})

    return cfg

  def save_artifacts(self):
    if not os.path.exists(self.output_dir):
      os.makedirs(self.output_dir)

    if len(self.autoencoders_dict) > 1:
      prefix = f"case-{self.case.get_name()}-multi-aes"
    else:
      prefix = f"case-{self.case.get_name()}-resid-{list(self.autoencoders_dict.values())[0].compression_size}"

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

    # The autoencoders have changed during the non-linear compression training, so we will save it.
    for ae_key, ae in self.autoencoders_dict.items():
      safe_ae_key = (ae_key
                     .replace('*', '.')
                     .replace('[', '.')
                     .replace(']', '.')
                     .replace("|", "."))
      prefix = f"case-{self.case.get_name()}-ae-{safe_ae_key}-size-{ae.compression_size}-final"
      ae.save(self.output_dir, prefix, self.wandb_run)
