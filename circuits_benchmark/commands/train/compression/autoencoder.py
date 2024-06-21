from argparse import Namespace

from argparse_dataclass import ArgumentParser

from circuits_benchmark.benchmark.benchmark_case import BenchmarkCase
from circuits_benchmark.benchmark.tracr_benchmark_case import TracrBenchmarkCase
from circuits_benchmark.commands.common_args import add_common_args
from circuits_benchmark.commands.train.compression.compression_training_utils import parse_d_model
from circuits_benchmark.training.compression.autencoder import AutoEncoder
from circuits_benchmark.training.compression.autoencoder_trainer import AutoEncoderTrainer
from circuits_benchmark.training.training_args import TrainingArgs
from circuits_benchmark.transformers.hooked_tracr_transformer import HookedTracrTransformer


def setup_args_parser(subparsers):
  parser = subparsers.add_parser("autoencoder")
  add_common_args(parser)

  parser.add_argument("--d-model", type=int, default=None,
                      help="The size of compressed residual stream.")
  parser.add_argument("--d-model-compression-ratio", type=float, default=None,
                      help="The size of compressed residual stream, expressed as a fraction of the original size.")

  parser.add_argument("--ae-layers", type=int, default=2,
                      help="The desired number of layers for the autoencoder.")
  parser.add_argument("--ae-first-hidden-layer-shape", type=str, default="wide", choices=["wide", "narrow"],
                      help="The desired shape for the first hidden layer of the autoencoder. Wide means larger than "
                           "the input layer. Narrow means smaller than the input layer.")


def train_autoencoder(case: BenchmarkCase, args: Namespace):
  """Trains an autoencoder to compress and decompress the residual stream space of a transformer."""
  assert isinstance(case, TracrBenchmarkCase), "Only TracrBenchmarkCase is supported for autoencoder training."
  tl_model: HookedTracrTransformer = case.get_hl_model()
  original_residual_stream_size = tl_model.cfg.d_model

  training_args, _ = ArgumentParser(TrainingArgs).parse_known_args(args.original_args)
  compression_size = parse_d_model(args, tl_model)

  autoencoder = AutoEncoder(original_residual_stream_size,
                            compression_size,
                            args.ae_layers,
                            args.ae_first_hidden_layer_shape)

  print(
    f" >>> Starting AutoEncoder training for {case} with residual stream compression size {compression_size}.")
  trainer = AutoEncoderTrainer(case, autoencoder, tl_model, training_args, output_dir=args.output_dir)
  final_metrics = trainer.train()
  print(f"\n >>> Final metrics for {case.get_name()}'s autoencoder with residual stream compression size {compression_size}: ")
  print(final_metrics)
