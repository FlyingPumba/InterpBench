import os
import pickle
import shutil
from argparse import Namespace
from copy import deepcopy
from dataclasses import dataclass
from typing import Tuple, Optional

import torch as t
from auto_circuit.data import PromptDataLoader, PromptDataset, PromptPairBatch
from auto_circuit.prune_algos.mask_gradient import mask_gradient_prune_scores
from auto_circuit.types import PruneScores, OutputSlice
from auto_circuit.utils.graph_utils import patchable_model
from auto_circuit.utils.tensor_ops import prune_scores_threshold

from circuits_benchmark.benchmark.benchmark_case import BenchmarkCase
from circuits_benchmark.commands.common_args import add_common_args, add_evaluation_common_ags
from circuits_benchmark.utils.auto_circuit_utils import build_circuit, build_normalized_scores
from circuits_benchmark.utils.circuit.circuit import Circuit
from circuits_benchmark.utils.circuit.circuit_eval import evaluate_hypothesis_circuit, CircuitEvalResult
from circuits_benchmark.utils.ll_model_loader.ll_model_loader import LLModelLoader
from circuits_benchmark.utils.project_paths import get_default_output_dir


@dataclass
class EAPConfig:
  threshold: Optional[float] = None
  seed: Optional[int] = 42
  edge_count: Optional[int] = None
  data_size: Optional[int] = 1000
  integrated_grad_steps: Optional[int] = None
  regression_loss_fn: Optional[str] = "mae"
  classification_loss_fn: Optional[str] = "kl_div"
  normalize_scores: Optional[bool] = False
  using_wandb: Optional[bool] = False
  output_dir: Optional[str] = get_default_output_dir()
  device: Optional[str] = "cpu"
  same_size: Optional[bool] = False
  include_mlp: Optional[bool] = False
  use_pos_embed: Optional[bool] = False
  weights: Optional[str] = None
  abs_value_threshold: Optional[bool] = False

  @staticmethod
  def from_args(args: Namespace) -> "EAPConfig":
    return EAPConfig(
      threshold=args.threshold,
      seed=int(args.seed),
      edge_count=args.edge_count,
      data_size=args.data_size,
      integrated_grad_steps=args.integrated_grad_steps,
      regression_loss_fn=args.regression_loss_fn,
      classification_loss_fn=args.classification_loss_fn,
      normalize_scores=args.normalize_scores,
      using_wandb=args.using_wandb,
      output_dir=args.output_dir,
      device=args.device,
      same_size=args.same_size,
      include_mlp=args.include_mlp,
      use_pos_embed=args.use_pos_embed,
      abs_value_threshold=args.abs_val_threshold
    )

class EAPRunner:
  def __init__(self,
               case: BenchmarkCase,
               config: EAPConfig | None = None,
               args: Namespace | None = None):
    self.case = case
    self.config = config
    self.args = deepcopy(args)

    if self.config is None:
      self.config = EAPConfig.from_args(args)

    assert self.config is not None

    self.data_size = self.config.data_size
    self.edge_count = self.config.edge_count
    self.threshold = self.config.threshold
    self.integrated_grad_steps = self.config.integrated_grad_steps
    self.regression_loss_fn = self.config.regression_loss_fn
    self.classification_loss_fn = self.config.classification_loss_fn
    self.normalize_scores = self.config.normalize_scores

    assert (self.edge_count is not None) ^ (self.threshold is not None), \
      "Either edge_count or threshold must be provided, but not both"

  def run_using_model_loader(self, ll_model_loader: LLModelLoader) -> Tuple[Circuit, CircuitEvalResult]:
    clean_dirname = self.prepare_output_dir(ll_model_loader)

    print(f"Running EAP evaluation for case {self.case.get_name()} ({str(ll_model_loader)})")
    print(f"Output directory: {clean_dirname}")

    hl_ll_corr, ll_model = ll_model_loader.load_ll_model_and_correspondence(
      device=self.config.device,
      output_dir=self.config.output_dir,
      same_size=self.config.same_size,
      # IOI specific args:
      eval=True,
      include_mlp=self.config.include_mlp,
      use_pos_embed=self.config.use_pos_embed
    )

    clean_dataset = self.case.get_clean_data(max_samples=self.config.data_size)
    corrupted_dataset = self.case.get_corrupted_data(max_samples=self.config.data_size)

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
    if self.config.using_wandb:
      import wandb
      algo_str = "eap" if self.integrated_grad_steps is None else f"integrated_grad_{self.integrated_grad_steps}" 
      wandb.init(
        project="circuit_discovery",
        group=f"{algo_str}_{self.case.get_name()}_{ll_model_loader.get_output_suffix()}",
        name=f"{self.config.threshold}" if self.config.threshold is not None else f"ec_{self.config.edge_count}",
      )
      wandb.save(f"{clean_dirname}/*", base_path=self.config.output_dir)

    return eap_circuit, result

  def run(
      self,
      tl_model: t.nn.Module,
      clean_inputs: t.Tensor,
      clean_outputs: t.Tensor,
      corrupted_inputs: t.Tensor,
      corrupted_outputs: t.Tensor
  ):
    slice_output: OutputSlice = "not_first_seq"  # This drops the first token from the output (e.g., BOS)
    if "ioi" in self.case.get_name():
      slice_output = "last_seq"  # Consider the last token as the output

    tl_model.to(self.config.device)
    auto_circuit_model = patchable_model(
      tl_model,
      factorized=True,
      slice_output=slice_output,
      separate_qkv=True,
      device=t.device(self.config.device),
    )

    dataset = PromptDataset(
      clean_inputs,
      corrupted_inputs,
      clean_outputs,
      corrupted_outputs,
    )
    train_loader = PromptDataLoader(dataset, seq_len=self.case.get_max_seq_len(), diverge_idx=0, batch_size=len(dataset))

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

    eap_args["answer_function"] = self.get_answer_function_for_case(
      tl_model,
      auto_circuit_model.out_slice  # type: ignore
    )

    attribution_scores: PruneScores = mask_gradient_prune_scores(**eap_args)
    if self.normalize_scores:
      attribution_scores = build_normalized_scores(attribution_scores)

    if self.edge_count is not None:
      # find the threshold for the top-k edges
      threshold = prune_scores_threshold(attribution_scores, self.edge_count).item()
      print(f"Threshold for top-{self.edge_count} edges: {threshold}")
    else:
      threshold = self.threshold

    eap_circuit = build_circuit(auto_circuit_model, attribution_scores, threshold, self.config.abs_value_threshold)
    eap_circuit.save(f"{self.config.output_dir}/final_circuit.pkl")

    return eap_circuit

  def get_answer_function_for_case(self,
                                   tl_model: t.nn.Module,
                                   out_slice: slice):
    if self.case.is_categorical():
      def loss_fn(logits: t.Tensor, batch: PromptPairBatch) -> t.Tensor:
        if batch.answers[out_slice].squeeze(dim=-1).shape == logits.shape:
          answers = batch.answers[out_slice].squeeze(dim=-1)
        else:
          answers= t.nn.functional.one_hot(batch.answers[out_slice].squeeze(dim=-1), num_classes=tl_model.cfg.d_vocab_out).float()
        log_probs = t.nn.functional.log_softmax(logits, dim=-1)
        kl = t.nn.functional.kl_div(log_probs, answers, reduction="batchmean", log_target=False)
        return kl
    
      # For categorical models we use as loss function the diff between the correct and wrong answers
      return "avg_diff" if self.classification_loss_fn == "avg_diff" else loss_fn
    else:
      # Auto-circuit assumes that all models are categorical, so we need to provide a custom loss function for
      # regression ones
      print(f"Using regression loss function: {self.regression_loss_fn}")

      def loss_fn(logits: t.Tensor, batch: PromptPairBatch) -> t.Tensor:
        print("logits:", logits.shape)
        print("answers:", batch.answers[out_slice].shape)
        if self.regression_loss_fn == "mse":
          return t.nn.functional.mse_loss(logits, batch.answers[out_slice]) - t.nn.functional.mse_loss(logits, batch.wrong_answers[out_slice])
        elif self.regression_loss_fn == "mae":
          return t.nn.functional.l1_loss(logits, batch.answers[out_slice])
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
      "--include-mlp", type=int, help="Evaluate group 'with_mlp'", default=1
    )
    parser.add_argument(
        "--use-pos-embed", action="store_true", help="Use positional embeddings"
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
    parser.add_argument("--classification-loss-fn", type=str, default="kl_div", choices=["kl_div", "avg_diff"])
    parser.add_argument("--normalize-scores", action="store_true",
                        help="Normalize the scores so that they all lie between 0 and 1.")
    parser.add_argument("--abs-val-threshold", action="store_true",
                        help="Use the absolute value of scores for thresholding.")

  def prepare_output_dir(self, ll_model_loader):
    if self.edge_count is not None:
      output_suffix = f"{ll_model_loader.get_output_suffix()}/edge_count_{self.edge_count}"
    else:
      output_suffix = f"{ll_model_loader.get_output_suffix()}/threshold_{self.threshold}"
      algorithm = "eap" if self.integrated_grad_steps is None else "integrated_grad"
    clean_dirname = f"{self.config.output_dir}/{algorithm}/{self.case.get_name()}/{output_suffix}"

    # remove everything in the directory
    if os.path.exists(clean_dirname):
      shutil.rmtree(clean_dirname)

    # mkdir
    os.makedirs(clean_dirname, exist_ok=True)

    return clean_dirname
