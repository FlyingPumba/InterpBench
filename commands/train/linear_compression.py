from argparse import Namespace

from argparse_dataclass import ArgumentParser

from benchmark.benchmark_case import BenchmarkCase
from commands.train.auto_compression import run_auto_compression_training
from training.compression.linear_compressed_tracr_transformer import LinearCompressedTracrTransformer, \
  linear_compression_initialization_options
from training.compression.linear_compressed_tracr_transformer_trainer import LinearCompressedTracrTransformerTrainer
from training.training_args import TrainingArgs
from utils.hooked_tracr_transformer import HookedTracrTransformer


def setup_args_parser(subparsers):
  parser = subparsers.add_parser("linear-compression")
  parser.add_argument("-i", "--indices", type=str, default=None,
                      help="A list of comma separated indices of the cases to run against. "
                           "If not specified, all cases will be run.")
  parser.add_argument("-f", "--force", action="store_true",
                      help="Force compilation of cases, even if they have already been compiled.")
  parser.add_argument("-o", "--output-dir", type=str, default="results",
                      help="The directory to save the results to.")

  parser.add_argument("--residual-stream-compression-size", type=str, default="auto",
                      help="A list of comma separated sizes for the compressed residual stream, or 'auto' to find the "
                           "optimal size.")
  parser.add_argument("--auto-compression-accuracy", type=float, default=0.95,
                      help="The desired test accuracy when using 'auto' compression size.")
  parser.add_argument("--linear-compression-initialization", type=str, default="linear",
                      choices=linear_compression_initialization_options,
                      help="The initialization method for the linear compression matrix.")


def run_single_linear_compression_training(case: BenchmarkCase,
                                           tl_model: HookedTracrTransformer,
                                           args: Namespace,
                                           compression_size: int):
  initialization = args.linear_compression_initialization
  training_args, _ = ArgumentParser(TrainingArgs).parse_known_args(args.original_args)

  print(f" >>> Starting linear compression for {case} with residual stream compression size {compression_size}.")
  compressed_tracr_transformer = LinearCompressedTracrTransformer(tl_model,
                                                                  int(compression_size),
                                                                  initialization=initialization,
                                                                  device=tl_model.device)
  training_args.wandb_name = None
  trainer = LinearCompressedTracrTransformerTrainer(case, compressed_tracr_transformer, training_args)
  final_metrics = trainer.train()
  print(f" >>> Final metrics for {case} with residual stream compression size {compression_size}: ")
  print(final_metrics)

  compressed_tracr_transformer.dump_compression_matrix(
    args.output_dir,
    f"case-{case.index_str}-resid-{str(compression_size)}-compression-matrix"
  )

  return final_metrics


def train_linear_compression(case: BenchmarkCase, args: Namespace):
  """Compresses the residual stream of a Tracr model using a linear compression."""
  tl_model: HookedTracrTransformer = case.load_tl_model()
  run_auto_compression_training(case, tl_model, args, run_single_linear_compression_training)
