from argparse import Namespace

from argparse_dataclass import ArgumentParser

from circuits_benchmark.benchmark.benchmark_case import BenchmarkCase
from circuits_benchmark.commands.common_args import add_common_args
from circuits_benchmark.commands.train.compression.compression_training_utils import parse_d_model
from circuits_benchmark.training.compression.linear_compressed_tracr_transformer import \
  LinearCompressedTracrTransformer, \
  linear_compression_initialization_options
from circuits_benchmark.training.compression.linear_compressed_tracr_transformer_trainer import \
  LinearCompressedTracrTransformerTrainer
from circuits_benchmark.training.training_args import TrainingArgs
from circuits_benchmark.transformers.hooked_tracr_transformer import HookedTracrTransformer


def setup_args_parser(subparsers):
  parser = subparsers.add_parser("linear-compression")
  add_common_args(parser)

  parser.add_argument("--d-model", type=int, default=None,
                      help="The size of compressed residual stream.")
  parser.add_argument("--d-model-compression-ratio", type=float, default=None,
                      help="The size of compressed residual stream, expressed as a fraction of the original size.")
  parser.add_argument("--linear-compression-initialization", type=str, default="linear",
                      choices=linear_compression_initialization_options,
                      help="The initialization method for the linear compression matrix.")


def train_linear_compression(case: BenchmarkCase, args: Namespace):
  """Compresses the residual stream of a Tracr model using a linear compression."""
  tl_model: HookedTracrTransformer = case.get_tl_model()
  training_args, _ = ArgumentParser(TrainingArgs).parse_known_args(args.original_args)

  compressed_d_model_size = parse_d_model(args, tl_model)

  initialization = args.linear_compression_initialization

  print(f" >>> Starting linear compression for {case} with residual stream compression size {compressed_d_model_size}.")
  compressed_tracr_transformer = LinearCompressedTracrTransformer(
    tl_model,
    compressed_d_model_size,
    initialization,
    tl_model.device)

  trainer = LinearCompressedTracrTransformerTrainer(case, tl_model, compressed_tracr_transformer, training_args,
                                                    output_dir=args.output_dir)
  final_metrics = trainer.train()
  print(f"\n >>> Final metrics for {case.get_index()} with residual stream compression size {compressed_d_model_size}:")
  print(final_metrics)

  return final_metrics
