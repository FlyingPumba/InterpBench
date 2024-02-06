from argparse import Namespace

from argparse_dataclass import ArgumentParser
from torch.nn import init

from benchmark.benchmark_case import BenchmarkCase
from commands.train.auto_compression import run_auto_compression_training
from training.compression.natural_compressed_tracr_transformer_trainer import NaturalCompressedTracrTransformerTrainer
from training.training_args import TrainingArgs
from utils.hooked_tracr_transformer import HookedTracrTransformer
from utils.project_paths import get_default_output_dir


def setup_args_parser(subparsers):
  parser = subparsers.add_parser("natural-compression")
  parser.add_argument("-i", "--indices", type=str, default=None,
                      help="A list of comma separated indices of the cases to run against. "
                           "If not specified, all cases will be run.")
  parser.add_argument("-f", "--force", action="store_true",
                      help="Force compilation of cases, even if they have already been compiled.")
  parser.add_argument("-o", "--output-dir", type=str, default=get_default_output_dir(),
                      help="The directory to save the results to.")

  parser.add_argument("--residual-stream-compression-size", type=str, default="auto",
                      help="A list of comma separated sizes for the compressed residual stream, or 'auto' to find the "
                           "optimal size.")
  parser.add_argument("--auto-compression-accuracy", type=float, default=0.95,
                      help="The desired test accuracy when using 'auto' compression size.")


def run_single_natural_compression_training(case: BenchmarkCase,
                                            tl_model: HookedTracrTransformer,
                                            args: Namespace,
                                            compression_size: int):
  training_args, _ = ArgumentParser(TrainingArgs).parse_known_args(args.original_args)

  print(f" >>> Starting natural compression for {case} with residual stream compression size {compression_size}.")
  new_tl_model = HookedTracrTransformer.from_hooked_tracr_transformer(
    tl_model,
    overwrite_cfg_dict={"d_model": compression_size},
    init_params_fn=lambda x: init.kaiming_uniform_(x) if len(x.shape) > 1 else init.normal_(x, std=0.02),
  )

  training_args.wandb_name = None
  trainer = NaturalCompressedTracrTransformerTrainer(case, tl_model, new_tl_model, training_args,
                                                     output_dir=args.output_dir)
  final_metrics = trainer.train()
  print(f" >>> Final metrics for {case} with residual stream compression size {compression_size}: ")
  print(final_metrics)

  return final_metrics


def train_natural_compression(case: BenchmarkCase, args: Namespace):
  """Trains a transformer from scratch, using the provided compression size."""
  tl_model: HookedTracrTransformer = case.load_tl_model()
  run_auto_compression_training(case, tl_model, args, run_single_natural_compression_training)
