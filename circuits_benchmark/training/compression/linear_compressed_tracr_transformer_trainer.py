from jaxtyping import Float
from torch import Tensor
from transformer_lens import ActivationCache, HookedTransformer, utils

from circuits_benchmark.benchmark.benchmark_case import BenchmarkCase
from circuits_benchmark.training.compression.causally_compressed_tracr_transformer_trainer import \
  CausallyCompressedTracrTransformerTrainer
from circuits_benchmark.training.compression.compression_train_loss_level import CompressionTrainLossLevel
from circuits_benchmark.training.compression.linear_compressed_tracr_transformer import LinearCompressedTracrTransformer
from circuits_benchmark.training.compression.residual_stream_mapper.linear_mapper import LinearMapper
from circuits_benchmark.training.compression.residual_stream_mapper.residual_stream_mapper import ResidualStreamMapper
from circuits_benchmark.training.training_args import TrainingArgs
from circuits_benchmark.transformers.hooked_tracr_transformer import HookedTracrTransformerBatchInput, \
  HookedTracrTransformer


class LinearCompressedTracrTransformerTrainer(CausallyCompressedTracrTransformerTrainer):
  def __init__(self,
               case: BenchmarkCase,
               original_model: HookedTracrTransformer,
               compressed_model: LinearCompressedTracrTransformer,
               args: TrainingArgs,
               train_loss_level: CompressionTrainLossLevel = "layer",
               output_dir: str | None = None):
    self.original_model = original_model
    self.compressed_model = compressed_model
    super().__init__(case,
                     list(compressed_model.parameters()),
                     args,
                     original_model.is_categorical(),
                     original_model.cfg.n_layers,
                     train_loss_level=train_loss_level,
                     output_dir=output_dir)

  def get_decoded_outputs_from_compressed_model(self, inputs: HookedTracrTransformerBatchInput) -> Tensor:
    return self.compressed_model(inputs, return_type="decoded")

  def get_logits_and_cache_from_compressed_model(
      self,
      inputs: HookedTracrTransformerBatchInput
  ) -> (Float[Tensor, "batch seq_len d_vocab"], ActivationCache):
    compressed_model_logits, compressed_model_cache = self.compressed_model.run_with_cache(inputs)

    if self.train_loss_level == "layer":
      # Decompress the residual streams of all layers except the last one, which we have already decompressed for using
      # the unembedding since TransformerLens does not have a hook for that.
      for layer in range(self.n_layers - 1):
        cache_key = utils.get_act_name("resid_post", layer)
        compressed_model_cache.cache_dict[cache_key] = self.get_residual_stream_mapper().decompress(
          compressed_model_cache.cache_dict[cache_key])

    elif self.train_loss_level == "component":
      # Decompress the residual streams outputted by the attention and mlp components of all layers
      for layer in range(self.n_layers):
        for component in ["attn", "mlp"]:
          hook_name = f"{component}_out"
          cache_key = utils.get_act_name(hook_name, layer)
          compressed_model_cache.cache_dict[cache_key] = self.get_residual_stream_mapper().decompress(
            compressed_model_cache.cache_dict[cache_key])

      for component in ["embed", "pos_embed"]:
        hook_name = f"hook_{component}"
        compressed_model_cache.cache_dict[hook_name] = self.get_residual_stream_mapper().decompress(
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