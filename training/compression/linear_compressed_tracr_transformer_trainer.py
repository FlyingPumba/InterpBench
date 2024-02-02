from jaxtyping import Float
from torch import Tensor
from transformer_lens import ActivationCache

from benchmark.benchmark_case import BenchmarkCase
from training.compression.compressed_tracr_transformer_trainer import CompressedTracrTransformerTrainer
from training.compression.linear_compressed_tracr_transformer import LinearCompressedTracrTransformer
from training.training_args import TrainingArgs
from utils.hooked_tracr_transformer import HookedTracrTransformerBatchInput


class LinearCompressedTracrTransformerTrainer(CompressedTracrTransformerTrainer):
  def __init__(self,
               case: BenchmarkCase,
               model: LinearCompressedTracrTransformer,
               args: TrainingArgs):
    super().__init__(case,
                     list(model.parameters()),
                     args,
                     model.get_tl_model().is_categorical(),
                     model.get_tl_model().cfg.n_layers)
    self.model = model

  def get_decoded_outputs_from_compressed_model(self, inputs: HookedTracrTransformerBatchInput) -> Tensor:
    return self.model(inputs, return_type="decoded")

  def get_logits_and_cache_from_compressed_model(
      self,
      inputs: HookedTracrTransformerBatchInput
  ) -> (Float[Tensor, "batch seq_len d_vocab"], ActivationCache):
    return self.model.run_with_cache(inputs)

  def get_logits_and_cache_from_original_model(
      self,
      inputs: HookedTracrTransformerBatchInput
  ) -> (Float[Tensor, "batch seq_len d_vocab"], ActivationCache):
    return self.model.run_with_cache_on_original(inputs)

  def build_wandb_name(self):
    return f"case-{self.case.index_str}-linear-resid-{self.model.residual_stream_compression_size}"
