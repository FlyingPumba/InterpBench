import dataclasses
import os

import torch as t
import transformer_lens.utils as utils
import wandb
from jaxtyping import Float
from torch import Tensor
from transformer_lens import ActivationCache, HookedTransformer

from benchmark.benchmark_case import BenchmarkCase
from training.compression.autencoder import AutoEncoder
from training.compression.autoencoder_trainer import AutoEncoderTrainer
from training.compression.compressed_tracr_transformer_trainer import CompressedTracrTransformerTrainer
from training.compression.residual_stream_mapper.autoencoder_mapper import AutoEncoderMapper
from training.compression.residual_stream_mapper.residual_stream_mapper import ResidualStreamMapper
from training.training_args import TrainingArgs
from utils.hooked_tracr_transformer import HookedTracrTransformer, HookedTracrTransformerBatchInput


class NonLinearCompressedTracrTransformerTrainer(CompressedTracrTransformerTrainer):
  def __init__(self, case: BenchmarkCase,
               old_tl_model: HookedTracrTransformer,
               new_tl_model: HookedTracrTransformer,
               autoencoder: AutoEncoder,
               args: TrainingArgs,
               output_dir: str | None = None,
               freeze_ae_weights: bool = False):
    self.old_tl_model: HookedTracrTransformer = old_tl_model
    self.new_tl_model: HookedTracrTransformer = new_tl_model
    self.autoencoder: AutoEncoder = autoencoder
    self.device = old_tl_model.device

    self.freeze_ae_weights = freeze_ae_weights
    if self.freeze_ae_weights:
      self.autoencoder.freeze_all_weights()
      self.autoencoder_trainer = None
    else:
      self.ae_epochs_per_training_step = 1
      # make a copy of the training args
      autoencoder_training_args = dataclasses.replace(args)
      self.autoencoder_trainer = AutoEncoderTrainer(case, self.autoencoder, self.old_tl_model,
                                                    autoencoder_training_args, output_dir=None)

    super().__init__(case,
                     list(new_tl_model.parameters()),
                     args,
                     old_tl_model.is_categorical(),
                     new_tl_model.cfg.n_layers,
                     output_dir=output_dir)

  def training_step(self, inputs) -> Float[Tensor, ""]:
    # train the autoencoder for some epochs before actually training the transformer
    avg_ae_train_loss = t.tensor(0.0, device=self.device)
    if not self.freeze_ae_weights:
      ae_train_losses = []
      for epoch in range(self.ae_epochs_per_training_step):
        for i, batch in enumerate(self.autoencoder_trainer.train_loader):
          ae_train_loss = self.autoencoder_trainer.training_step(batch)
          ae_train_losses.append(ae_train_loss)

      avg_ae_train_loss = t.mean(t.stack(ae_train_losses))

      if self.use_wandb:
        # log average train loss and test mse
        test_inputs = next(iter(self.autoencoder_trainer.test_loader))
        outputs = self.autoencoder(test_inputs)

        ae_test_mse = t.nn.functional.mse_loss(test_inputs, outputs).item()
        wandb.log({
          "ae_test_mse": ae_test_mse,
          "ae_train_loss": avg_ae_train_loss
        }, step=self.step)

    train_loss = super().training_step(inputs)
    loss = train_loss + avg_ae_train_loss

    return loss

  def get_decoded_outputs_from_compressed_model(self, inputs: HookedTracrTransformerBatchInput) -> Tensor:
    return self.new_tl_model(inputs, return_type="decoded")

  def get_logits_and_cache_from_compressed_model(
      self,
      inputs: HookedTracrTransformerBatchInput
  ) -> (Float[Tensor, "batch seq_len d_vocab"], ActivationCache):
    compressed_model_logits, compressed_model_cache = self.new_tl_model.run_with_cache(inputs)

    for layer in range(self.n_layers):
      cache_key = utils.get_act_name("resid_post", layer)
      compressed_model_cache.cache_dict[cache_key] = self.get_residual_stream_mapper().decompress(
        compressed_model_cache.cache_dict[cache_key])

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

  def get_residual_stream_mapper(self) -> ResidualStreamMapper:
    return AutoEncoderMapper(self.autoencoder)

  def build_wandb_name(self):
    return f"case-{self.case.get_index()}-non-linear-resid-{self.autoencoder.compression_size}"

  def save_artifacts(self):
    if not os.path.exists(self.output_dir):
      os.makedirs(self.output_dir)

    prefix = f"case-{self.case.get_index()}-resid-{self.autoencoder.compression_size}"

    # save the weights of the model using state_dict
    weights_path = os.path.join(self.output_dir, f"{prefix}-non-linear-compression-weights.pt")
    t.save(self.autoencoder.state_dict(), weights_path)

    # save the entire model
    model_path = os.path.join(self.output_dir, f"{prefix}-non-linearly-compressed-tracr-transformer.pt")
    t.save(self.autoencoder, model_path)

    if self.wandb_run is not None:
      # save the files as artifacts to wandb
      artifact = wandb.Artifact(f"{prefix}-non-linearly-compressed-tracr-transformer", type="model")
      artifact.add_file(weights_path)
      artifact.add_file(model_path)
      self.wandb_run.log_artifact(artifact)

    if not self.freeze_ae_weights:
      # The autoencoder has changed during the non-linear compression training, so we will save it.
      prefix = f"case-{self.case.get_index()}-resid-{self.autoencoder.compression_size}-final"
      self.autoencoder.save(self.output_dir, prefix, self.wandb_run)