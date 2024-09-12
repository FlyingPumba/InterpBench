import dataclasses
from argparse import Namespace

import pandas as pd
import wandb
from argparse_dataclass import ArgumentParser
from iit.model_pairs.ll_model import LLModel

from circuits_benchmark.benchmark.benchmark_case import BenchmarkCase
from circuits_benchmark.benchmark.tracr_benchmark_case import TracrBenchmarkCase
from circuits_benchmark.commands.common_args import add_common_args
from circuits_benchmark.commands.train.compression.compression_training_utils import parse_d_model, parse_d_head
from circuits_benchmark.metrics.iia import evaluate_iia_on_all_ablation_types
from circuits_benchmark.training.compression.autencoder import AutoEncoder
from circuits_benchmark.training.compression.non_linear_compressed_tracr_transformer_trainer import \
    NonLinearCompressedTracrTransformerTrainer
from circuits_benchmark.training.training_args import TrainingArgs
from circuits_benchmark.transformers.hooked_tracr_transformer import HookedTracrTransformer
from circuits_benchmark.utils.init_functions import wang_init_method


def setup_args_parser(subparsers):
  parser = subparsers.add_parser("non-linear-compression")
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

  parser.add_argument("--ae-path", type=str, default=None,
                      help="Path to trained AutoEncoder model.")

  parser.add_argument("--ae-layers", type=int, default=2,
                      help="The desired number of layers for the autoencoder.")
  parser.add_argument("--ae-first-hidden-layer-shape", type=str, default="wide", choices=["wide", "narrow"],
                      help="The desired shape for the first hidden layer of the autoencoder. Wide means larger than "
                           "the input layer. Narrow means smaller than the input layer.")

  parser.add_argument("--ae-epochs", type=int, default=70,
                      help="The number of epochs to use for initial autoencoder training.")
  parser.add_argument("--ae-batch-size", type=int, default=2**12,
                      help="The batch size to use for autoencoder training.")
  parser.add_argument("--ae-lr-start", type=float, default=0.01,
                      help="The initial learning rate to use for autoencoder training.")
  parser.add_argument("--ae-max-train-samples", type=int, default=2048,
                      help="The maximum number of training samples to use for autoencoder training.")

  parser.add_argument("--ae-max-training-epochs", type=int, default=15,
                      help="The max number of epochs to use for training when autoencoder weights are not frozen.")
  parser.add_argument("--ae-desired-test-mse", type=float, default=1e-3,
                      help="The desired test mean squared error for the autoencoder.")
  parser.add_argument("--ae-train-loss-weight", type=float, default=0.5,
                      help="The weight for the autoencoder training loss in the total loss.")


def train_non_linear_compression(case: BenchmarkCase, args: Namespace):
  """Compresses the residual stream of a Tracr model using a linear compression."""
  assert isinstance(case, TracrBenchmarkCase), "Only TracrBenchmarkCase is supported for autoencoder training."
  hl_model: HookedTracrTransformer = case.get_hl_model()
  original_d_model_size = hl_model.cfg.d_model
  original_d_head_size = hl_model.cfg.d_head

  training_args, _ = ArgumentParser(TrainingArgs).parse_known_args(args.original_args)

  compressed_d_model_size = parse_d_model(args, hl_model)
  compressed_d_head_size = parse_d_head(args, hl_model)

  # Get LL model with compressed dimensions
  ll_model = case.get_ll_model(
    overwrite_cfg_dict={
      "d_model": compressed_d_model_size,
      "d_head": compressed_d_head_size,
      "d_mlp": compressed_d_model_size * 4
    },
    same_size=True
  )

  # reset params
  init_fn=wang_init_method(hl_model.cfg.n_layers, compressed_d_model_size)
  for name, param in ll_model.named_parameters():
    init_fn(param)

  # Set up autoencoders
  autoencoders_dict = {}
  autoencoders_dict["blocks.*.hook_mlp_out"] = AutoEncoder(original_d_model_size,
                                                           compressed_d_model_size,
                                                           args.ae_layers,
                                                           args.ae_first_hidden_layer_shape)
  for layer in range(hl_model.cfg.n_layers):
    for head in range(hl_model.cfg.n_heads):
      autoencoders_dict[f"blocks.{layer}.attn.hook_result[{head}]"] = AutoEncoder(original_d_model_size,
                                                                                  compressed_d_model_size,
                                                                                  args.ae_layers,
                                                                                  args.ae_first_hidden_layer_shape)

  ae_training_args = dataclasses.replace(training_args,
                                         wandb_project=None,
                                         wandb_name=None,
                                         epochs=args.ae_epochs,
                                         batch_size=args.ae_batch_size,
                                         lr_start=args.ae_lr_start,
                                         max_train_samples=args.ae_max_train_samples)

  trainer = NonLinearCompressedTracrTransformerTrainer(case,
                                                       LLModel(model=hl_model),
                                                       ll_model,
                                                       autoencoders_dict,
                                                       training_args,
                                                       output_dir=args.output_dir,
                                                       ae_training_args=ae_training_args,
                                                       ae_desired_test_mse=args.ae_desired_test_mse,
                                                       ae_train_loss_weight=args.ae_train_loss_weight)
  print(
    f" >>> Starting transformer training for {case} non-linear compressed resid of size {compressed_d_model_size} and "
    f"compressed head size {compressed_d_head_size}.")
  final_metrics = trainer.train(finish_wandb_run=False)
  print(f"\n >>> Final metrics for {case.get_name()}'s non-linear compressed transformer with resid size {compressed_d_model_size} and "
        f"compressed head size {compressed_d_head_size}:")
  print(final_metrics)

  iia_eval_results = evaluate_iia_on_all_ablation_types(case, LLModel(model=hl_model), ll_model, trainer.test_dataset)
  print(f" >>> IIA evaluation results:")
  for node_str, result in iia_eval_results.items():
    print(result)

  # Save iia_eval_results as csv
  iia_eval_results_df = pd.DataFrame(iia_eval_results).T
  iia_eval_results_csv_path = f"{args.output_dir}/iia_eval_results.csv"
  iia_eval_results_df.to_csv(iia_eval_results_csv_path, index=False)

  if trainer.wandb_run is not None:
    # save the files as artifacts to wandb
    prefix = f"case-{case.get_name()}-multi-aes"
    artifact = wandb.Artifact(f"{prefix}-iia-evaluation", type="csv")
    artifact.add_file(iia_eval_results_csv_path)
    trainer.wandb_run.log_artifact(artifact)

    wandb.finish()

  return final_metrics
