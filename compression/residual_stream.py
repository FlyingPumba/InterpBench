import typing
from argparse import Namespace
from typing import Literal

from argparse_dataclass import _add_dataclass_options, ArgumentParser

from benchmark.benchmark_case import BenchmarkCase
from compression.compressed_tracr_transformer import CompressedTracrTransformer
from compression.compressed_tracr_transformer_trainer import CompressionTrainingArgs, CompressedTracrTransformerTrainer
from utils.hooked_tracr_transformer import HookedTracrTransformer

residual_stream_compression_options_type = Literal["linear"]
residual_stream_compression_options = list(typing.get_args(residual_stream_compression_options_type))


def setup_compression_training_args_for_parser(parser):
  parser.add_argument("--compress-residual", type=str, choices=residual_stream_compression_options,
                              default=None,
                              help="Compress residual stream in the Tracr models.")
  parser.add_argument("--residual-stream-compression-size", type=str, default="auto",
                              help="The size of the compressed residual stream. Choose 'auto' to find the optimal size.")


def compress(case: BenchmarkCase,
             tl_model: HookedTracrTransformer,
             compression_type: residual_stream_compression_options_type,
             residual_stream_compression_size: int | Literal["auto"],
             args: Namespace):
  """Compresses the residual stream of a Tracr model.

  Tracr models can be sparse and inefficient because they reserve an orthogonal subspace of the residual stream for
  each s-op. This function forces different levels of superposition by applying a gradent-descent-based compression
  algorithm. This is useful for studying the effect of superposition and make Tracr models more efficient and realistic.
  """

  assert (residual_stream_compression_size == "auto" or
          (0 < int(residual_stream_compression_size) <= tl_model.cfg.d_model)), \
    f"Invalid residual stream compression size: {residual_stream_compression_size}. " \
    f"Must be between 0 and {tl_model.cfg.d_model} or 'auto'."

  if compression_type == "linear":
    compress_linear(case, tl_model, int(residual_stream_compression_size), args)
  else:
    raise ValueError(f"Unknown compression type: {compression_type}")


def compress_linear(case: BenchmarkCase,
                    tl_model: HookedTracrTransformer,
                    residual_stream_compression_size: int | Literal["auto"],
                    args: Namespace):
  """Compresses the residual stream of a Tracr model using a linear compression."""
  assert residual_stream_compression_size != "auto", "Auto compression size not supported yet."

  compressed_tracr_transformer = CompressedTracrTransformer(tl_model,
                                                            residual_stream_compression_size,
                                                            device=tl_model.device)
  training_args, _ = ArgumentParser(CompressionTrainingArgs).parse_known_args(args.original_args)
  dataset = case.get_clean_data(count=training_args.train_data_size)
  trainer = CompressedTracrTransformerTrainer(training_args, compressed_tracr_transformer, dataset)
  trainer.train()

  tl_model.reset_hooks(including_permanent=True)

