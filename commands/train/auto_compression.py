from argparse import Namespace

from argparse_dataclass import ArgumentParser

from benchmark.benchmark_case import BenchmarkCase
from commands.train.compression_training_utils import parse_compression_size
from training.training_args import TrainingArgs
from utils.hooked_tracr_transformer import HookedTracrTransformer


def run_auto_compression_training(case: BenchmarkCase,
                                  tl_model: HookedTracrTransformer,
                                  args: Namespace,
                                  run_single_compression_training_fn):
  original_residual_stream_size = tl_model.cfg.d_model
  compression_size = parse_compression_size(args, tl_model)

  training_args, _ = ArgumentParser(TrainingArgs).parse_known_args(args.original_args)
  original_wandb_name = training_args.wandb_name

  if compression_size != "auto":
    for compression_size in compression_size:
      run_single_compression_training_fn(case, tl_model, args, compression_size)
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

    # Halve the residual stream size until we get a test accuracy below the desired one.
    while current_compression_size > 0:
      if original_wandb_name is not None:
        # add a suffix to the wandb name to indicate the current compression size
        args.wandb_name = f"{original_wandb_name}-size-{current_compression_size}"
      final_metrics = run_single_compression_training_fn(case, tl_model, args, current_compression_size)

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

        if original_wandb_name is not None:
          # add a suffix to the wandb name to indicate the current compression size
          args.wandb_name = f"{original_wandb_name}-size-{current_compression_size}"
        final_metrics = run_single_compression_training_fn(case, tl_model, args, current_compression_size)

        if final_metrics["test_accuracy"] > desired_test_accuracy:
          upper_bound = current_compression_size
        else:
          lower_bound = current_compression_size + 1

      best_compression_size = upper_bound

    print(f" >>> Best residual stream compression size for {case} is {best_compression_size}.")