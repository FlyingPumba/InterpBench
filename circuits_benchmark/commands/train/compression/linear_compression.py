from argparse import Namespace

from circuits_benchmark.benchmark.benchmark_case import BenchmarkCase
from circuits_benchmark.commands.common_args import add_common_args
from circuits_benchmark.commands.train.compression.auto_compression import run_auto_compression_training
from circuits_benchmark.training.compression.compression_train_loss_level import compression_train_loss_level_options
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

  parser.add_argument("--residual-stream-compression-size", type=str, default="auto",
                      help="A list of comma separated sizes for the compressed residual stream, or 'auto' to find the "
                           "optimal size.")
  parser.add_argument("--auto-compression-accuracy", type=float, default=0.95,
                      help="The desired test accuracy when using 'auto' compression size.")
  parser.add_argument("--train-loss", type=str, default="layer", choices=compression_train_loss_level_options,
                      help="The train loss level for the compression training.")
  parser.add_argument("--linear-compression-initialization", type=str, default="linear",
                      choices=linear_compression_initialization_options,
                      help="The initialization method for the linear compression matrix.")


def run_single_linear_compression_training(case: BenchmarkCase,
                                           tl_model: HookedTracrTransformer,
                                           args: Namespace,
                                           training_args: TrainingArgs,
                                           compression_size: int):
  initialization = args.linear_compression_initialization

  print(f" >>> Starting linear compression for {case} with residual stream compression size {compression_size}.")
  compressed_tracr_transformer = LinearCompressedTracrTransformer(
    tl_model,
    int(compression_size),
    initialization,
    tl_model.device)

  trainer = LinearCompressedTracrTransformerTrainer(case, tl_model, compressed_tracr_transformer, training_args,
                                                    train_loss_level=args.train_loss,
                                                    output_dir=args.output_dir)
  final_metrics = trainer.train()
  print(f" >>> Final metrics for {case} with residual stream compression size {compression_size}: ")
  print(final_metrics)

  return final_metrics


def train_linear_compression(case: BenchmarkCase, args: Namespace):
  """Compresses the residual stream of a Tracr model using a linear compression."""
  tl_model: HookedTracrTransformer = case.get_tl_model()
  run_auto_compression_training(case, tl_model, args, run_single_linear_compression_training)
