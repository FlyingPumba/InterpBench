import dataclasses
import re
from argparse import Namespace
from math import ceil

import pandas as pd
import torch as t
from argparse_dataclass import ArgumentParser

from circuits_benchmark.benchmark.benchmark_case import BenchmarkCase
from circuits_benchmark.commands.common_args import add_common_args
from circuits_benchmark.metrics.iia import evaluate_iia_on_all_ablation_types
from circuits_benchmark.training.compression.attention_trainer import AttentionTrainer
from circuits_benchmark.training.compression.autencoder import AutoEncoder
from circuits_benchmark.training.compression.autoencoder_trainer import AutoEncoderTrainer
from circuits_benchmark.training.compression.mlp_trainer import MLPTrainer
from circuits_benchmark.training.training_args import TrainingArgs
from circuits_benchmark.transformers.hooked_tracr_transformer import HookedTracrTransformer
from circuits_benchmark.utils.init_functions import wang_init_method


def setup_args_parser(subparsers):
  parser = subparsers.add_parser("individual-components-compression")
  add_common_args(parser)

  parser.add_argument("--ae-layers", type=int, default=2,
                      help="The desired number of layers for the autoencoder.")
  parser.add_argument("--ae-first-hidden-layer-shape", type=str, default="wide", choices=["wide", "narrow"],
                      help="The desired shape for the first hidden layer of the autoencoder. Wide means larger than "
                           "the input layer. Narrow means smaller than the input layer.")

  parser.add_argument("--ae-epochs", type=int, default=70,
                      help="The number of epochs to use for initial autoencoder training.")
  parser.add_argument("--ae-batch-size", type=int, default=2**12,
                      help="The batch size to use for initial autoencoder training.")
  parser.add_argument("--ae-lr-start", type=float, default=0.01,
                      help="The number of epochs to use for initial autoencoder training.")

  parser.add_argument("--ae-desired-test-mse", type=float, default=1e-3,
                      help="The desired test mean squared error for the autoencoder.")



def train_individual_components_compression(case: BenchmarkCase, args: Namespace):
  original_model: HookedTracrTransformer = case.get_tl_model()
  training_args, _ = ArgumentParser(TrainingArgs).parse_known_args(args.original_args)

  original_d_model_size = original_model.cfg.d_model
  original_d_head_size = original_model.cfg.d_head

  compressed_d_model_size = ceil(original_d_model_size * 2 / 3)
  compressed_d_head_size = ceil(original_d_head_size / 2)

  compressed_model = HookedTracrTransformer.from_hooked_tracr_transformer(
    original_model,
    overwrite_cfg_dict={
      "d_model": compressed_d_model_size,
      "d_head": compressed_d_head_size,
    },
    init_params_fn=wang_init_method(original_model.cfg.n_layers, compressed_d_model_size),
  )

  # Set up autoencoders for compression training
  autoencoders_dict = {}
  autoencoders_dict["blocks\\..\\.hook_mlp_out"] = AutoEncoder(original_d_model_size,
                                                           compressed_d_model_size,
                                                           args.ae_layers,
                                                           args.ae_first_hidden_layer_shape)
  for layer in range(original_model.cfg.n_layers):
    for head in range(original_model.cfg.n_heads):
      autoencoders_dict[f"blocks.{layer}.attn.hook_result[{head}]"] = AutoEncoder(original_d_model_size,
                                                                                  compressed_d_model_size,
                                                                                  args.ae_layers,
                                                                                  args.ae_first_hidden_layer_shape)

  ae_training_args = dataclasses.replace(training_args,
                                         wandb_project=None,
                                         wandb_name=None,
                                         epochs=args.ae_epochs,
                                         batch_size=args.ae_batch_size,
                                         lr_start=args.ae_lr_start)
  data = case.get_clean_data(count=training_args.train_data_size)

  with t.no_grad():
    _, activations_cache = original_model.run_with_cache(data.get_inputs())

  autoencoder_trainers_dict = {}
  for ae_key, ae in autoencoders_dict.items():
    ae_trainer = AutoEncoderTrainer(case, ae, original_model, ae_training_args,
                                    train_loss_level="intervention",
                                    hook_name_filter_for_input_activations=ae_key,
                                    output_dir=args.output_dir,
                                    data_activations=activations_cache)
    autoencoder_trainers_dict[ae_key] = ae_trainer

  train_autoencoders(autoencoder_trainers_dict, ae_training_args.epochs, args.ae_desired_test_mse)

  for layer in range(original_model.cfg.n_layers):
    input_hook_name = f"blocks.{layer}.hook_resid_pre"

    for head in range(original_model.cfg.n_heads):
      # Train attention head
      output_hook_name = f"blocks.{layer}.attn.hook_result"
      ae = find_ae_for_hook(autoencoders_dict, output_hook_name, head_index=head)
      attn_trainer = AttentionTrainer(case,
                                      compressed_model.blocks[layer].attn,
                                      original_model,
                                      ae,
                                      input_hook_name,
                                      output_hook_name,
                                      activations_cache,
                                      training_args,
                                      head_index=head,
                                      output_dir=args.output_dir)
      print(f"Training attention head {layer} - {head}")
      final_metrics = attn_trainer.train()
      print(final_metrics)


    # Train MLP
    output_hook_name = f"blocks.{layer}.hook_mlp_out"
    ae = find_ae_for_hook(autoencoders_dict, output_hook_name)
    mlp_trainer = MLPTrainer(case,
                             compressed_model.blocks[layer].mlp,
                             original_model,
                             ae,
                             input_hook_name,
                             output_hook_name,
                             activations_cache,
                             training_args,
                             output_dir=args.output_dir)
    print(f"Training MLP {layer}")
    final_metrics = mlp_trainer.train()
    print(final_metrics)

  iia_eval_results = evaluate_iia_on_all_ablation_types(case, original_model, compressed_model)
  print(f" >>> IIA evaluation results:")
  for node_str, result in iia_eval_results.items():
    base_model_effect = result["base_model_effect_resample_ablation"]
    hypothesis_model_effect = result["hypothesis_model_effect_resample_ablation"]
    effect_diff = abs(base_model_effect - hypothesis_model_effect)
    print(f"{node_str} -> {effect_diff:.3f}")

  # Save iia_eval_results as csv
  iia_eval_results_df = pd.DataFrame(iia_eval_results).T
  iia_eval_results_csv_path = f"{args.output_dir}/iia_eval_results.csv"
  iia_eval_results_df.to_csv(iia_eval_results_csv_path, index=False)


def train_autoencoders(autoencoder_trainers_dict, max_epochs: int, ae_desired_test_mse: float):
  avg_ae_train_loss = None

  for ae_key, ae_trainer in autoencoder_trainers_dict.items():
      ae_trainer.compute_test_metrics()
      ae_training_epoch = 0
      while (ae_trainer.test_metrics["test_mse"] > ae_desired_test_mse and
             ae_training_epoch < max_epochs):
        ae_train_losses = []
        for i, batch in enumerate(ae_trainer.train_loader):
          ae_train_loss = ae_trainer.training_step(batch)
          ae_train_losses.append(ae_train_loss)

        avg_ae_train_loss = t.mean(t.stack(ae_train_losses))

        ae_trainer.compute_test_metrics()
        ae_training_epoch += 1

      print(f"AutoEncoder {ae_key} trained for {ae_training_epoch} epochs, and achieved train loss of {avg_ae_train_loss}.")
      print(f"AutoEncoder {ae_key} test metrics: {ae_trainer.test_metrics}")

def find_ae_for_hook(autoencoders_dict, hook_name: str, head_index: int|None=None) -> AutoEncoder:
  node_name = hook_name
  if head_index is not None:
    node_name += f"[{head_index}]"

  for ae_key, ae in autoencoders_dict.items():
    string = f"^{ae_key}$".replace("[", "\\[").replace("]", "\\]")
    regex = re.compile(string)
    if regex.match(node_name):
      return ae

  raise ValueError(f"No AutoEncoder found for hook {hook_name} and head_index {head_index}.")