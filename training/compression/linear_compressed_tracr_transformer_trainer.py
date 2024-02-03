from jaxtyping import Float
from torch import Tensor
from transformer_lens import ActivationCache, HookedTransformer

from benchmark.benchmark_case import BenchmarkCase
from training.compression.compressed_tracr_transformer_trainer import CompressedTracrTransformerTrainer
from training.compression.linear_compressed_tracr_transformer import LinearCompressedTracrTransformer
from training.training_args import TrainingArgs
from utils.hooked_tracr_transformer import HookedTracrTransformerBatchInput, HookedTracrTransformer


class LinearCompressedTracrTransformerTrainer(CompressedTracrTransformerTrainer):
  def __init__(self,
               case: BenchmarkCase,
               original_model: HookedTracrTransformer,
               compressed_model: LinearCompressedTracrTransformer,
               args: TrainingArgs):
    super().__init__(case,
                     list(compressed_model.parameters()),
                     args,
                     original_model.is_categorical(),
                     original_model.cfg.n_layers)
    self.original_model = original_model
    self.compressed_model = compressed_model

  def get_decoded_outputs_from_compressed_model(self, inputs: HookedTracrTransformerBatchInput) -> Tensor:
    return self.compressed_model(inputs, return_type="decoded")

  def get_logits_and_cache_from_compressed_model(
      self,
      inputs: HookedTracrTransformerBatchInput
  ) -> (Float[Tensor, "batch seq_len d_vocab"], ActivationCache):
    return self.compressed_model.run_with_cache(inputs)

  def get_logits_and_cache_from_original_model(
      self,
      inputs: HookedTracrTransformerBatchInput
  ) -> (Float[Tensor, "batch seq_len d_vocab"], ActivationCache):
    return self.original_model.run_with_cache(inputs)

  def get_original_model(self) -> HookedTransformer:
    return self.original_model

  def get_compressed_model(self) -> HookedTransformer:
    return self.compressed_model

  def build_wandb_name(self):
    return f"case-{self.case.get_index()}-linear-resid-{self.compressed_model.residual_stream_compression_size}"
