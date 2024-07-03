from transformer_lens import HookedTransformer

from circuits_benchmark.benchmark.benchmark_case import BenchmarkCase
from circuits_benchmark.training.compression.activation_mapper.activation_mapper import ActivationMapper
from circuits_benchmark.training.compression.activation_mapper.linear_mapper import LinearMapper
from circuits_benchmark.training.compression.causally_compressed_tracr_transformer_trainer import \
  CausallyCompressedTracrTransformerTrainer
from circuits_benchmark.training.compression.linear_compressed_tracr_transformer import LinearCompressedTracrTransformer
from circuits_benchmark.training.training_args import TrainingArgs
from circuits_benchmark.transformers.hooked_tracr_transformer import HookedTracrTransformer


class LinearCompressedTracrTransformerTrainer(CausallyCompressedTracrTransformerTrainer):
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

  def get_original_model(self) -> HookedTransformer:
    return self.original_model

  def get_compressed_model(self) -> HookedTransformer:
    return self.compressed_model

  def get_activation_mapper(self) -> ActivationMapper | None:
    return LinearMapper(self.compressed_model.W_compress)

  def build_wandb_name(self):
    return f"case-{self.case.get_name()}-linear-resid-{self.compressed_model.residual_stream_compression_size}"

  def get_wandb_tags(self):
    tags = super().get_wandb_tags()
    tags.append("linear-compression-trainer")
    return tags

  def save_artifacts(self):
    prefix = f"case-{self.case.get_name()}-resid-{self.compressed_model.residual_stream_compression_size}"
    self.compressed_model.save(self.output_dir, prefix, self.wandb_run)