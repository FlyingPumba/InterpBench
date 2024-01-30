import typing
from argparse import Namespace
from functools import partial
from typing import Literal

from argparse_dataclass import ArgumentParser
from torch import nn
from torch.nn import init
from transformer_lens import HookedTransformer, HookedTransformerConfig

from benchmark.benchmark_case import BenchmarkCase
from compression.autencoder import AutoEncoder
from compression.autoencoder_trainer import AutoEncoderTrainer
from compression.linear_compressed_tracr_transformer import LinearCompressedTracrTransformer, \
  linear_compression_initialization_options
from compression.linear_compressed_tracr_transformer_trainer import CompressionTrainingArgs, LinearCompressedTracrTransformerTrainer
from compression.non_linear_compressed_tracr_transformer_trainer import NonLinearCompressedTracrTransformerTrainer
from utils.hooked_tracr_transformer import HookedTracrTransformer

residual_stream_compression_options_type = Literal["linear", "nonlinear"]
residual_stream_compression_options = list(typing.get_args(residual_stream_compression_options_type))


def setup_compression_training_args_for_parser(parser):
  parser.add_argument("--compress-residual", type=str, choices=residual_stream_compression_options, default=None,
                      help="Compress residual stream in the Tracr models.")
  parser.add_argument("--residual-stream-compression-size", type=str, default="auto",
                      help="A list of comma separated sizes for the compressed residual stream, or 'auto' to find the "
                           "optimal size.")
  parser.add_argument("--auto-compression-accuracy", type=float, default=0.95,
                      help="The desired test accuracy when using 'auto' compression size.")
  parser.add_argument("--linear-compression-initialization", type=str, default="linear",
                      choices=linear_compression_initialization_options,
                      help="The initialization method for the linear compression matrix.")
  parser.add_argument("-o", "--output-dir", type=str, default="results",
                      help="The directory to save the results to.")


def compress(case: BenchmarkCase,
             tl_model: HookedTracrTransformer,
             compression_type: residual_stream_compression_options_type,
             args: Namespace):
  """Compresses the residual stream of a Tracr model.

  Tracr models can be sparse and inefficient because they reserve an orthogonal subspace of the residual stream for
  each s-op. This function forces different levels of superposition by applying a gradent-descent-based compression
  algorithm. This is useful for studying the effect of superposition and make Tracr models more efficient and realistic.
  """
  if compression_type == "linear":
    compress_linear(case, tl_model, args)
  elif compression_type == "nonlinear":
    compress_non_linear(case, tl_model, args)
  else:
    raise ValueError(f"Unknown compression type: {compression_type}")


def parse_compression_size(args, tl_model: HookedTracrTransformer):
  compression_size = args.residual_stream_compression_size
  if compression_size == "auto":
    return compression_size

  # separate by commas and convert to integers
  compression_size = [int(size.strip()) for size in compression_size.split(",")]

  assert all(0 < size <= tl_model.cfg.d_model for size in compression_size), \
    f"Invalid residual stream compression size: {str(compression_size)}. " \
      f"All sizes in a comma separated list must be between 0 and {tl_model.cfg.d_model}."

  assert len(compression_size) > 0, "Must specify at least one residual stream compression size."

  return compression_size


def compress_linear(case: BenchmarkCase,
                    tl_model: HookedTracrTransformer,
                    args: Namespace):
  """Compresses the residual stream of a Tracr model using a linear compression."""
  compression_size = parse_compression_size(args, tl_model)
  initialization = args.linear_compression_initialization

  training_args, _ = ArgumentParser(CompressionTrainingArgs).parse_known_args(args.original_args)
  original_residual_stream_size = tl_model.cfg.d_model

  if compression_size != "auto":
    for compression_size in compression_size:
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
  else:
    desired_test_accuracy = args.auto_compression_accuracy
    assert 0 < desired_test_accuracy <= 1, f"Invalid desired test accuracy: {desired_test_accuracy}. " \
                                           f"Must be between 0 and 1."

    # The "auto" mode of compression is a binary search for the optimal residual stream compression size: We want the
    # smallest residual stream size that achieves a desired test accuracy in the final_metrics.
    # We start by compressing the residual stream to half its original size. We keep halving the size until we get a
    # test accuracy below the desired one. Then we do a binary search between the last size that was above the desired
    # accuracy and the last size that was below the desired accuracy.

    print(f" >>> Starting auto linear compression for {case}.")
    print(f" >>> Original residual stream size is {original_residual_stream_size}.")
    print(f" >>> Desired test accuracy is {desired_test_accuracy}.")

    current_compression_size = original_residual_stream_size // 2
    best_compression_size = original_residual_stream_size
    compressed_tracr_transformer = None

    # Halve the residual stream size until we get a test accuracy below the desired one.
    while current_compression_size > 0:
      compressed_tracr_transformer = LinearCompressedTracrTransformer(tl_model,
                                                                      current_compression_size,
                                                                      initialization=initialization,
                                                                      device=tl_model.device)
      training_args.wandb_name = None
      trainer = LinearCompressedTracrTransformerTrainer(case, compressed_tracr_transformer, training_args)
      final_metrics = trainer.train()

      if final_metrics["test_accuracy"] > desired_test_accuracy:
        best_compression_size = current_compression_size
        current_compression_size = current_compression_size // 2
      else:
        break

    # Do a binary search between the last size that was above the desired accuracy and the last size that was below the
    # desired accuracy.
    if current_compression_size > 0:
      lower_bound = current_compression_size
      upper_bound = best_compression_size

      while lower_bound < upper_bound:
        current_compression_size = (lower_bound + upper_bound) // 2
        compressed_tracr_transformer = LinearCompressedTracrTransformer(tl_model,
                                                                        current_compression_size,
                                                                        initialization=initialization,
                                                                        device=tl_model.device)
        training_args.wandb_name = None
        trainer = LinearCompressedTracrTransformerTrainer(case, compressed_tracr_transformer, training_args)
        final_metrics = trainer.train()

        if final_metrics["test_accuracy"] > desired_test_accuracy:
          upper_bound = current_compression_size
        else:
          lower_bound = current_compression_size + 1

      best_compression_size = upper_bound

    print(f" >>> Best residual stream compression size for {case} is {best_compression_size}.")

    compressed_tracr_transformer.dump_compression_matrix(
      args.output_dir,
      f"case-{case.index_str}-resid-{str(compression_size)}-compression-matrix"
    )


def compress_non_linear(case: BenchmarkCase,
                        tl_model: HookedTracrTransformer,
                        args: Namespace):
  """Compresses the residual stream of a Tracr model using a non-linear compression."""
  compression_size = parse_compression_size(args, tl_model)

  training_args, _ = ArgumentParser(CompressionTrainingArgs).parse_known_args(args.original_args)
  original_residual_stream_size = tl_model.cfg.d_model

  if compression_size != "auto":
    for compression_size in compression_size:
      autoencoder_compression_layers = 2
      autoencoder = AutoEncoder(original_residual_stream_size, compression_size, autoencoder_compression_layers)

      print(f" >>> Starting non-linear compression for {case} with residual stream compression size {compression_size}.")
      training_args.wandb_name = None
      trainer = AutoEncoderTrainer(case, autoencoder, tl_model, training_args)
      final_metrics = trainer.train()
      print(f" >>> Final metrics for {case}'s autoencoder with residual stream compression size {compression_size}: ")
      print(final_metrics)

      new_tl_model = HookedTracrTransformer.from_hooked_tracr_transformer(
        tl_model,
        overwrite_cfg_dict={"d_model": compression_size},
        init_params_fn=lambda x: init.kaiming_uniform_(x) if len(x.shape) > 1 else init.normal_(x, std=0.02),
      )

      autoencoder.freeze_all_weights()
      new_tl_model.unfreeze_all_weights()

      print(f" >>> Starting transformer training for {case} non-linear compressed resid of size {compression_size}.")
      training_args.wandb_name = None
      trainer = NonLinearCompressedTracrTransformerTrainer(case, tl_model, new_tl_model, autoencoder, training_args)
      final_metrics = trainer.train()
      print(f" >>> Final metrics for {case}'s non-linear compressed transformer with resid size {compression_size}: ")
      print(final_metrics)

      # compressed_tracr_transformer.dump_compression_matrix(
      #   args.output_dir,
      #   f"case-{case.index_str}-resid-{str(compression_size)}-compression-matrix"
      # )
  else:
    raise NotImplementedError("Non-linear compression with auto compression size is not implemented yet.")