from jaxtyping import Float
from torch import Tensor
from transformer_lens import ActivationCache

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
               args: TrainingArgs):
    super().__init__(case,
                     new_tl_model.parameters(),
                     args,
                     old_tl_model.is_categorical(),
                     new_tl_model.cfg.n_layers)

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
    self.new_tl_model.run_with_cache(inputs)

  def get_logits_and_cache_from_original_model(
      self,
      inputs: HookedTracrTransformerBatchInput
  ) -> (Float[Tensor, "batch seq_len d_vocab"], ActivationCache):
    self.old_tl_model.run_with_cache(inputs)

  def build_wandb_name(self):
    return f"case-{self.case.index_str}-non-linear-resid-{self.autoencoder.compression_size}"
