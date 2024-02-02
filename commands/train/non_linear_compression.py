from argparse import Namespace

from argparse_dataclass import ArgumentParser
from torch.nn import init

from benchmark.benchmark_case import BenchmarkCase
from commands.train.auto_compression import run_auto_compression_training
from training.compression.autencoder import AutoEncoder
from training.compression.non_linear_compressed_tracr_transformer_trainer import \
  NonLinearCompressedTracrTransformerTrainer
from training.training_args import TrainingArgs
from utils.hooked_tracr_transformer import HookedTracrTransformer


def setup_args_parser(subparsers):
  parser = subparsers.add_parser("non-linear-compression")
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
  parser.add_argument("--ae-path", type=str, required=True,
                      help="Path to trained AutoEncoder model.")
  parser.add_argument("--ae-layers", type=int, default=2,
                      help="The desired number of layers for the autoencoder.")


def run_single_non_linear_compression_training(case: BenchmarkCase,
                                           tl_model: HookedTracrTransformer,
                                           args: Namespace,
                                           compression_size: int):
  original_residual_stream_size = tl_model.cfg.d_model
  training_args, _ = ArgumentParser(TrainingArgs).parse_known_args(args.original_args)

  new_tl_model = HookedTracrTransformer.from_hooked_tracr_transformer(
    tl_model,
    overwrite_cfg_dict={"d_model": compression_size},
    init_params_fn=lambda x: init.kaiming_uniform_(x) if len(x.shape) > 1 else init.normal_(x, std=0.02),
  )

  # Load AutoEncoder model
  autoencoder = AutoEncoder(original_residual_stream_size, compression_size, args.ae_layers)
  autoencoder.load_weights_from_file(args.ae_path)

  autoencoder.freeze_all_weights()
  new_tl_model.unfreeze_all_weights()

  print(f" >>> Starting transformer training for {case} non-linear compressed resid of size {compression_size}.")
  trainer = NonLinearCompressedTracrTransformerTrainer(case, tl_model, new_tl_model, autoencoder, training_args)
  final_metrics = trainer.train()
  print(f" >>> Final metrics for {case}'s non-linear compressed transformer with resid size {compression_size}: ")
  print(final_metrics)

  # compressed_tracr_transformer.dump_compression_matrix(
  #   args.output_dir,
  #   f"case-{case.index_str}-resid-{str(compression_size)}-compression-matrix"
  # )

  return final_metrics


def train_non_linear_compression(case: BenchmarkCase, args: Namespace):
  """Compresses the residual stream of a Tracr model using a linear compression."""
  tl_model: HookedTracrTransformer = case.load_tl_model()
  run_auto_compression_training(case, tl_model, args, run_single_non_linear_compression_training)
