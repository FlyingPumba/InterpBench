import os

import torch as t
import transformer_lens.utils as utils
import wandb
from jaxtyping import Float
from torch import Tensor
from transformer_lens import ActivationCache, HookedTransformer

from benchmark.benchmark_case import BenchmarkCase
from training.compression.autencoder import AutoEncoder
from training.compression.compressed_tracr_transformer_trainer import CompressedTracrTransformerTrainer
from training.training_args import TrainingArgs
from utils.hooked_tracr_transformer import HookedTracrTransformer, HookedTracrTransformerBatchInput


class NonLinearCompressedTracrTransformerTrainer(CompressedTracrTransformerTrainer):
  def __init__(self, case: BenchmarkCase,
               old_tl_model: HookedTracrTransformer,
               new_tl_model: HookedTracrTransformer,
               autoencoder: AutoEncoder,
               args: TrainingArgs,
               output_dir: str | None = None):
    super().__init__(case,
                     list(new_tl_model.parameters()),
                     args,
                     old_tl_model.is_categorical(),
                     new_tl_model.cfg.n_layers,
                     output_dir=output_dir)
    self.old_tl_model: HookedTracrTransformer = old_tl_model
    self.new_tl_model: HookedTracrTransformer = new_tl_model
    self.autoencoder: AutoEncoder = autoencoder
    self.device = old_tl_model.device

  def update_params(self, loss: Float[Tensor, ""]):
    loss.backward(retain_graph=True)
    self.optimizer.step()
    self.lr_scheduler.step()

  def get_decoded_outputs_from_compressed_model(self, inputs: HookedTracrTransformerBatchInput) -> Tensor:
    return self.new_tl_model(inputs, return_type="decoded")

  def get_logits_and_cache_from_compressed_model(
      self,
      inputs: HookedTracrTransformerBatchInput
  ) -> (Float[Tensor, "batch seq_len d_vocab"], ActivationCache):
    compressed_model_logits, compressed_model_cache = self.new_tl_model.run_with_cache(inputs)

    for layer in range(self.n_layers):
      cache_key = utils.get_act_name("resid_post", layer)
      compressed_model_cache.cache_dict[cache_key] = self.autoencoder.decoder(
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