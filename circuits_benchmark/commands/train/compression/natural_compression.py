from argparse import Namespace

from argparse_dataclass import ArgumentParser
from torch.nn import init

from circuits_benchmark.benchmark.benchmark_case import BenchmarkCase
from circuits_benchmark.commands.common_args import add_common_args
from circuits_benchmark.commands.train.compression.compression_training_utils import parse_d_model, parse_d_head
from circuits_benchmark.training.compression.natural_compressed_tracr_transformer_trainer import \
  NaturalCompressedTracrTransformerTrainer
from circuits_benchmark.training.training_args import TrainingArgs
from circuits_benchmark.transformers.hooked_tracr_transformer import HookedTracrTransformer


def setup_args_parser(subparsers):
  parser = subparsers.add_parser("natural-compression")
  add_common_args(parser)

  parser.add_argument("--d-model", type=int, default=None,
                      help="The size of compressed residual stream.")
  parser.add_argument("--d-model-compression-ratio", type=float, default=None,
                      help="The size of compressed residual stream, expressed as a fraction of the original size.")
  parser.add_argument("--d-head", type=int, default=None,
                      help="The size of compressed internal head dimension.")
  parser.add_argument("--d-head-compression-ratio", type=float, default=None,
                      help="The size of compressed internal head dimension, expressed as a fraction of the original "
                           "size.")


def train_natural_compression(case: BenchmarkCase, args: Namespace):
  """Trains a transformer from scratch, using the provided compression size."""
  tl_model: HookedTracrTransformer = case.get_tl_model()
  training_args, _ = ArgumentParser(TrainingArgs).parse_known_args(args.original_args)

  compressed_d_model_size = parse_d_model(args, tl_model)
  compressed_d_head_size = parse_d_head(args, tl_model)

  print(f" >>> Starting natural compression for {case} with residual stream compression size {compressed_d_model_size} "
        f"and internal head compression size {compressed_d_head_size}.")
  new_tl_model = HookedTracrTransformer.from_hooked_tracr_transformer(
    tl_model,
    overwrite_cfg_dict={
      "d_model": compressed_d_model_size,
      "d_head": compressed_d_head_size,
    },
    init_params_fn=lambda x: init.kaiming_uniform_(x) if len(x.shape) > 1 else init.normal_(x, std=0.02),
  )

  trainer = NaturalCompressedTracrTransformerTrainer(case, tl_model, new_tl_model, training_args,
                                                     output_dir=args.output_dir)
  final_metrics = trainer.train()
  print(f" >>> Final metrics for {case} with residual stream compression size {compressed_d_model_size} "
        f"and internal head compression size {compressed_d_head_size}:")
  print(final_metrics)

  return final_metrics
