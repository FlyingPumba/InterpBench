import os
import pickle
import shutil
from argparse import Namespace
from typing import Tuple

import torch as t
from auto_circuit.data import PromptDataLoader, PromptDataset, PromptPairBatch
from auto_circuit.prune_algos.mask_gradient import mask_gradient_prune_scores
from auto_circuit.types import PruneScores
from auto_circuit.utils.graph_utils import patchable_model
from auto_circuit.utils.tensor_ops import prune_scores_threshold

from circuits_benchmark.benchmark.benchmark_case import BenchmarkCase
from circuits_benchmark.commands.common_args import add_common_args, add_evaluation_common_ags
from circuits_benchmark.utils.auto_circuit_utils import build_circuit
from circuits_benchmark.utils.circuit.circuit import Circuit
from circuits_benchmark.utils.circuit.circuit_eval import evaluate_hypothesis_circuit, CircuitEvalResult
from circuits_benchmark.utils.ll_model_loader.ll_model_loader import LLModelLoader


class EAPRunner:
  def __init__(self, case: BenchmarkCase, args: Namespace):
    self.case = case
    self.args = args
    self.data_size = args.data_size
    self.edge_count = args.edge_count
    self.threshold = args.threshold
    self.integrated_grad_steps = args.integrated_grad_steps
    self.regression_loss_fn = args.regression_loss_fn
    self.normalize_scores = args.normalize_scores

    assert (self.edge_count is not None) ^ (self.threshold is not None), \
      "Either edge_count or threshold must be provided, but not both"

  def run_using_model_loader(self, ll_model_loader: LLModelLoader) -> Tuple[Circuit, CircuitEvalResult]:
    clean_dirname = self.prepare_output_dir(ll_model_loader)

    print(f"Running EAP evaluation for case {self.case.get_name()} ({str(ll_model_loader)})")
    print(f"Output directory: {clean_dirname}")

    hl_ll_corr, ll_model = ll_model_loader.load_ll_model_and_correspondence(
      load_from_wandb=self.args.load_from_wandb,
      device=self.args.device,
      output_dir=self.args.output_dir,
      same_size=self.args.same_size,
      # IOI specific args:
      eval=True,
      include_mlp=self.args.include_mlp,
      use_pos_embed=False
    )

    clean_dataset = self.case.get_clean_data(max_samples=self.args.data_size)
    corrupted_dataset = self.case.get_corrupted_data(max_samples=self.args.data_size)

    clean_outputs = clean_dataset.get_targets()
    corrupted_outputs = corrupted_dataset.get_targets()
    if self.case.is_categorical():
      if isinstance(clean_outputs, list):
        clean_outputs = [o.argmax(dim=-1).unsqueeze(dim=-1) for o in clean_outputs]
        corrupted_outputs = [o.argmax(dim=-1).unsqueeze(dim=-1) for o in corrupted_outputs]
      elif isinstance(clean_outputs, t.Tensor):
        clean_outputs = clean_outputs.argmax(dim=-1).unsqueeze(dim=-1)
        corrupted_outputs = corrupted_outputs.argmax(dim=-1).unsqueeze(dim=-1)
      else:
        raise ValueError(f"Unknown output type: {type(clean_outputs)}")

    eap_circuit = self.run(
      ll_model,
      clean_dataset.get_inputs(),
      clean_outputs,
      corrupted_dataset.get_inputs(),
      corrupted_outputs,
    )

    print("hl_ll_corr:", hl_ll_corr)
    hl_ll_corr.save(f"{clean_dirname}/hl_ll_corr.pkl")

    print("Calculating FPR and TPR")
    result = evaluate_hypothesis_circuit(
      eap_circuit,
      ll_model,
      hl_ll_corr,
      self.case,
      use_embeddings=False,
    )

    # save the result
    with open(f"{clean_dirname}/result.txt", "w") as f:
      f.write(str(result))

    pickle.dump(result, open(f"{clean_dirname}/result.pkl", "wb"))
    print(f"Saved result to {clean_dirname}/result.txt and {clean_dirname}/result.pkl")
    if self.args.using_wandb:
      import wandb

      wandb.init(
        project="circuit_discovery",
        group=f"eap_{self.case.get_name()}_{self.args.weights}",
        name=f"{self.args.threshold}",
      )
      wandb.save(f"{clean_dirname}/*", base_path=self.args.output_dir)

    return eap_circuit, result

  def run(
      self,
      tl_model: t.nn.Module,
      clean_inputs: t.Tensor,
      clean_outputs: t.Tensor,
      corrupted_inputs: t.Tensor,
      corrupted_outputs: t.Tensor
  ):
    tl_model.to(self.args.device)
    auto_circuit_model = patchable_model(
      tl_model,
      factorized=True,
      slice_output=None,
      separate_qkv=True,
      device=self.args.device,
    )

    dataset = PromptDataset(
      clean_inputs,
      corrupted_inputs,
      clean_outputs,
      corrupted_outputs,
    )
    train_loader = PromptDataLoader(dataset, seq_len=None, diverge_idx=0)

    eap_args = {
      "model": auto_circuit_model,
      "dataloader": train_loader,
      "official_edges": None,
      "grad_function": "logit",
      "mask_val": None,
      "integrated_grad_samples": None,
    }

    if self.integrated_grad_steps is not None:
      eap_args["integrated_grad_samples"] = self.integrated_grad_steps
    else:
      eap_args["mask_val"] = 0.0

    eap_args["answer_function"] = self.get_answer_function_for_case()

    attribution_scores: PruneScores = mask_gradient_prune_scores(**eap_args)
    if self.normalize_scores:
      attribution_scores = self.build_normalized_scores(attribution_scores)

    if self.edge_count is not None:
      # find the threshold for the top-k edges
      threshold = prune_scores_threshold(attribution_scores, self.edge_count).item()
      print(f"Threshold for top-{self.edge_count} edges: {threshold}")
    else:
      threshold = self.threshold

    eap_circuit = build_circuit(auto_circuit_model, attribution_scores, threshold)
    eap_circuit.save(f"{self.args.output_dir}/final_circuit.pkl")

    return eap_circuit

  def build_normalized_scores(self, attribution_scores: PruneScores) -> PruneScores:
    """Normalize the scores so that they all lie between 0 and 1."""
    max_score = max(scores.max() for scores in attribution_scores.values())
    min_score = min(scores.min() for scores in attribution_scores.values())

    normalized_scores = attribution_scores.copy()
    for module_name, scores in normalized_scores.items():
      normalized_scores[module_name] = (normalized_scores[module_name] - min_score) / (max_score - min_score)

    return normalized_scores

  def get_answer_function_for_case(self):
    if self.case.is_categorical():
      # For categorical models we use as loss function the diff between the correct and wrong answers
      return "avg_diff"
    else:
      # Auto-circuit assumes that all models are categorical, so we need to provide a custom loss function for
      # regression ones
      print(f"Using regression loss function: {self.regression_loss_fn}")

      def loss_fn(logits: t.Tensor, batch: PromptPairBatch) -> t.Tensor:
        if self.regression_loss_fn == "mse":
          return t.nn.functional.mse_loss(logits, batch.answers) - t.nn.functional.mse_loss(logits, batch.wrong_answers)
        elif self.regression_loss_fn == "mae":
          return t.nn.functional.l1_loss(logits, batch.answers)
        else:
          raise ValueError(f"Unknown regression loss function: {self.regression_loss_fn}")

      return loss_fn

  @staticmethod
  def setup_subparser(subparsers):
    parser = subparsers.add_parser("eap")
    EAPRunner.add_args_to_parser(parser)

  @staticmethod
  def add_args_to_parser(parser):
    add_common_args(parser)
    add_evaluation_common_ags(parser)

    parser.add_argument(
      "--include-mlp", action="store_true", help="Evaluate group 'with_mlp'"
    )
    parser.add_argument(
        "-wandb", "--using_wandb", action="store_true", help="Use wandb"
    )
    parser.add_argument("--edge-count", type=int, default=None,
                        help="Number of edges to keep in the final circuit")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Threshold of effect to keep an edge in the final circuit")
    parser.add_argument("--data-size", type=int, default=1000, help="Number of samples to use")
    parser.add_argument("--integrated-grad-steps", type=int, default=None,
                        help="Number of samples for integrated grad. If None, this is not used.")
    parser.add_argument("--regression-loss-fn", type=str, default="mae",
                        choices=["mse", "mae"], help="Loss function to use for regression models.")
    parser.add_argument("--normalize-scores", action="store_true",
                        help="Normalize the scores so that they all lie between 0 and 1.")

  def prepare_output_dir(self, ll_model_loader):
    if self.edge_count is not None:
      output_suffix = f"{ll_model_loader.get_output_suffix()}/edge_count_{self.edge_count}"
    else:
      output_suffix = f"{ll_model_loader.get_output_suffix()}/threshold_{self.threshold}"

    clean_dirname = f"{self.args.output_dir}/eap/{self.case.get_name()}/{output_suffix}"

    # remove everything in the directory
    if os.path.exists(clean_dirname):
      shutil.rmtree(clean_dirname)

    # mkdir
    os.makedirs(clean_dirname, exist_ok=True)

    return clean_dirname
