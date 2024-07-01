from argparse import Namespace, ArgumentParser

import torch as t
from auto_circuit.data import PromptDataLoader, PromptDataset, PromptPairBatch
from auto_circuit.prune_algos.mask_gradient import mask_gradient_prune_scores
from auto_circuit.types import PruneScores
from auto_circuit.utils.graph_utils import patchable_model
from auto_circuit.utils.tensor_ops import prune_scores_threshold

from circuits_benchmark.benchmark.benchmark_case import BenchmarkCase
from circuits_benchmark.benchmark.case_dataset import CaseDataset
from circuits_benchmark.commands.common_args import add_common_args
from circuits_benchmark.utils.auto_circuit_utils import build_circuit


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

  def run_on_tracr_model(self):
    tl_model = self.case.get_tl_model()
    clean_dataset = self.case.get_clean_data(count=self.data_size)
    corrupted_dataset = self.case.get_corrupted_data(count=self.data_size)

    return self.run(tl_model, clean_dataset, corrupted_dataset)

  def run(self,
          tl_model: t.nn.Module,
          clean_dataset: CaseDataset,
          corrupted_dataset: CaseDataset):
    tl_model.to(self.args.device)
    auto_circuit_model = patchable_model(
      tl_model,
      factorized=True,
      slice_output=None,
      separate_qkv=True,
      device=self.args.device,
    )

    # remove from inputs the rows that have the same expected output
    clean_raw_inputs = clean_dataset.get_inputs()
    corrupted_raw_inputs = corrupted_dataset.get_inputs()
    clean_expected_outputs = clean_dataset.get_correct_outputs()
    corrupted_expected_outputs = corrupted_dataset.get_correct_outputs()

    idxs_to_remove = []
    for i in range(len(clean_expected_outputs)):
      if clean_expected_outputs[i] == corrupted_expected_outputs[i]:
        idxs_to_remove.append(i)

    clean_inputs = [clean_raw_inputs[i] for i in range(len(clean_raw_inputs)) if i not in idxs_to_remove]
    corrupted_inputs = [corrupted_raw_inputs[i] for i in range(len(corrupted_raw_inputs)) if i not in idxs_to_remove]

    # Convert inputs to tensors using tracr encoder
    clean_inputs = tl_model.map_tracr_input_to_tl_input(clean_inputs)
    corrupted_inputs = tl_model.map_tracr_input_to_tl_input(corrupted_inputs)

    # Use as correct and wrong answers the output of tracr model on the filtered inputs
    with t.no_grad():
      if tl_model.is_categorical():
        clean_outputs = tl_model(clean_inputs).argmax(dim=-1).unsqueeze(dim=-1)
        corrupted_outputs = tl_model(corrupted_inputs).argmax(dim=-1).unsqueeze(dim=-1)
      else:
        clean_outputs = tl_model(clean_inputs)
        corrupted_outputs = tl_model(corrupted_inputs)

    dataset = PromptDataset(
      clean_inputs,
      corrupted_inputs,
      clean_outputs,
      corrupted_outputs,
    )
    train_loader = PromptDataLoader(dataset, seq_len=self.case.get_max_seq_len(), diverge_idx=0)

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

    if tl_model.is_categorical():
      # For categorical models we use as loss function the diff between the correct and wrong answers
      eap_args["answer_function"] = "avg_diff"
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

      eap_args["answer_function"] = loss_fn

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

  @staticmethod
  def setup_subparser(subparsers):
    parser = subparsers.add_parser("eap")
    EAPRunner.add_args_to_parser(parser)

  @staticmethod
  def add_args_to_parser(parser):
    add_common_args(parser)

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

  @classmethod
  def make_default_runner(cls, task: str):
    parser = ArgumentParser()
    cls.add_args_to_parser(parser)
    args = parser.parse_args([])
    return cls(task, args)
