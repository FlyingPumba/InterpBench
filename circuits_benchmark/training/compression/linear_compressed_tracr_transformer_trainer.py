from jaxtyping import Float
from torch import Tensor
from transformer_lens import ActivationCache, HookedTransformer, utils

from circuits_benchmark.benchmark.benchmark_case import BenchmarkCase
from circuits_benchmark.training.compression.compressed_tracr_transformer_trainer import CompressedTracrTransformerTrainer
from circuits_benchmark.training.compression.linear_compressed_tracr_transformer import LinearCompressedTracrTransformer
from circuits_benchmark.training.compression.residual_stream_mapper.linear_mapper import LinearMapper
from circuits_benchmark.training.compression.residual_stream_mapper.residual_stream_mapper import ResidualStreamMapper
from circuits_benchmark.training.training_args import TrainingArgs
from circuits_benchmark.transformers.hooked_tracr_transformer import HookedTracrTransformerBatchInput, HookedTracrTransformer


class LinearCompressedTracrTransformerTrainer(CompressedTracrTransformerTrainer):
  def __init__(self,
               case: BenchmarkCase,
               original_model: HookedTracrTransformer,
               compressed_model: LinearCompressedTracrTransformer,
               args: TrainingArgs,
               output_dir: str | None = None):
    self.original_model = original_model
    self.compressed_model = compressed_model
    super().__init__(case,
                     list(compressed_model.parameters()),
                     args,
                     original_model.is_categorical(),
                     original_model.cfg.n_layers,
                     output_dir=output_dir)

  def get_decoded_outputs_from_compressed_model(self, inputs: HookedTracrTransformerBatchInput) -> Tensor:
    return self.compressed_model(inputs, return_type="decoded")

  def get_logits_and_cache_from_compressed_model(
      self,
      inputs: HookedTracrTransformerBatchInput
  ) -> (Float[Tensor, "batch seq_len d_vocab"], ActivationCache):
    compressed_model_logits, compressed_model_cache = self.compressed_model.run_with_cache(inputs)

    # Decompress the residual streams of all layers except the last one, which we have already decompressed for using
    # the unembedding since TransformerLens does not have a hook for that.
    for layer in range(self.n_layers - 1):
      cache_key = utils.get_act_name("resid_post", layer)
      compressed_model_cache.cache_dict[cache_key] = self.get_residual_stream_mapper().decompress(
        compressed_model_cache.cache_dict[cache_key])

    return compressed_model_logits, compressed_model_cache

  def get_logits_and_cache_from_original_model(
      self,
      inputs: HookedTracrTransformerBatchInput
  ) -> (Float[Tensor, "batch seq_len d_vocab"], ActivationCache):
    return self.original_model.run_with_cache(inputs)

  def get_original_model(self) -> HookedTransformer:
    return self.original_model

  def get_compressed_model(self) -> HookedTransformer:
    return self.compressed_model

  def get_residual_stream_mapper(self) -> ResidualStreamMapper:
    return LinearMapper(self.compressed_model.W_compress)

  def build_wandb_name(self):
    return f"case-{self.case.get_index()}-linear-resid-{self.compressed_model.residual_stream_compression_size}"

  def get_wandb_tags(self):
    tags = super().get_wandb_tags()
    tags.append("linear-compression-trainer")
    return tags

  def save_artifacts(self):
    prefix = f"case-{self.case.get_index()}-resid-{self.compressed_model.residual_stream_compression_size}"
    self.compressed_model.save(self.output_dir, prefix, self.wandb_run)