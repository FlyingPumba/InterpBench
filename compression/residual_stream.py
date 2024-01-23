import typing
from typing import Literal

from benchmark.benchmark_case import BenchmarkCase
from compression.compressed_tracr_transformer import CompressedTracrTransformer
from compression.compressed_tracr_transformer_trainer import CompressionTrainingArgs, CompressedTracrTransformerTrainer
from utils.hooked_tracr_transformer import HookedTracrTransformer

residual_stream_compression_options_type = Literal["linear"]
residual_stream_compression_options = list(typing.get_args(residual_stream_compression_options_type))


def compress(case: BenchmarkCase,
             tl_model: HookedTracrTransformer,
             compression_type: residual_stream_compression_options_type,
             residual_stream_compression_size: int | Literal["auto"]):
  """Compresses the residual stream of a Tracr model.

  Tracr models can be sparse and inefficient because they reserve an orthogonal subspace of the residual stream for
  each s-op. This function forces different levels of superposition by applying a gradent-descent-based compression
  algorithm. This is useful for studying the effect of superposition and make Tracr models more efficient and realistic.
  """

  assert (residual_stream_compression_size == "auto" or
          (0 < residual_stream_compression_size < tl_model.cfg.d_model)), \
    f"Invalid residual stream compression size: {residual_stream_compression_size}. " \
    f"Must be between 0 and {tl_model.cfg.d_model} or 'auto'."

  if compression_type == "linear":
    compress_linear(case, tl_model, residual_stream_compression_size)
  else:
    raise ValueError(f"Unknown compression type: {compression_type}")


def compress_linear(case: BenchmarkCase,
                    tl_model: HookedTracrTransformer,
                    residual_stream_compression_size: int | Literal["auto"]):
  """Compresses the residual stream of a Tracr model using a linear compression."""
  assert residual_stream_compression_size != "auto", "Auto compression size not supported yet."

  compressed_tracr_transformer = CompressedTracrTransformer(tl_model,
                                                            residual_stream_compression_size)
  training_args = CompressionTrainingArgs()
  dataset = case.get_clean_data(count=300)
  trainer = CompressedTracrTransformerTrainer(training_args, compressed_tracr_transformer, dataset)
  trainer.train()

  tl_model.reset_hooks(including_permanent=True)

