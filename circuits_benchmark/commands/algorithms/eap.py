from argparse import Namespace, ArgumentParser

import torch as t
from auto_circuit.data import PromptDataLoader, PromptDataset
from auto_circuit.prune_algos.edge_attribution_patching import edge_attribution_patching_prune_scores
from auto_circuit.types import PruneScores
from auto_circuit.utils.graph_utils import patchable_model
from auto_circuit.utils.patchable_model import PatchableModel
from auto_circuit.utils.tensor_ops import prune_scores_threshold
from jaxtyping import Float

from acdc.TLACDCCorrespondence import TLACDCCorrespondence
from circuits_benchmark.benchmark.benchmark_case import BenchmarkCase
from circuits_benchmark.commands.common_args import add_common_args
from circuits_benchmark.transformers.acdc_circuit_builder import build_acdc_circuit
from circuits_benchmark.transformers.circuit import Circuit
from circuits_benchmark.transformers.circuit_node import CircuitNode
from circuits_benchmark.utils.circuits_comparison import calculate_fpr_and_tpr


class EAPRunner:
  def __init__(self, case: BenchmarkCase, args: Namespace):
    self.case = case
    self.args = args
    self.edge_count = args.top_k

  def run(self):
    tl_model = self.case.get_tl_model()
    assert tl_model.is_categorical(), \
      "EAP only works with categorical models for now, due to the way it computes the answer diff for loss."

    auto_circuit_model = patchable_model(
      tl_model,
      factorized=True,
      slice_output="last_seq",
      separate_qkv=True,
      device=self.args.device,
    )

    clean_dataset = self.case.get_clean_data(count=100)
    corrupt_dataset = self.case.get_corrupted_data(count=100)

    dataset = PromptDataset(
      tl_model.map_tracr_input_to_tl_input(clean_dataset.get_inputs()),
      tl_model.map_tracr_input_to_tl_input(corrupt_dataset.get_inputs()),
      tl_model(clean_dataset.get_inputs()).argmax(dim=-1),
      tl_model(corrupt_dataset.get_inputs()).argmax(dim=-1),
    )
    train_loader = PromptDataLoader(dataset, seq_len=self.case.get_max_seq_len(), diverge_idx=0)

    attribution_scores: PruneScores = edge_attribution_patching_prune_scores(
      model=auto_circuit_model,
      dataloader=train_loader,
      official_edges=None,
      answer_diff=tl_model.is_categorical(),
    )

    # find the threshold for the top-k edges and build circuit using that threshold
    threshold = prune_scores_threshold(attribution_scores, self.edge_count)
    eap_circuit = self.build_final_circuit(auto_circuit_model, attribution_scores, threshold)
    eap_circuit.save(f"{self.args.output_dir}/final_circuit.pkl")

    print("Calculating FPR and TPR for threshold", threshold)
    full_corr = TLACDCCorrespondence.setup_from_model(tl_model, use_pos_embed=True)
    full_circuit = build_acdc_circuit(full_corr)
    tracr_hl_circuit, tracr_ll_circuit, alignment = self.case.get_tracr_circuit(granularity="acdc_hooks")
    result = calculate_fpr_and_tpr(eap_circuit, tracr_ll_circuit, full_circuit, verbose=True)

    return eap_circuit, result

  def build_final_circuit(self,
                          model: PatchableModel,
                          attribution_scores: PruneScores,
                          threshold: Float[t.Tensor, ""]) -> Circuit:
    circuit = Circuit()

    for edge in model.edges:
      src_node = edge.src
      dst_node = edge.dest
      score = attribution_scores[dst_node.module_name][edge.patch_idx]

      if score > threshold:
        from_node = CircuitNode(src_node.module_name, src_node.head_idx)
        to_node = CircuitNode(dst_node.module_name, dst_node.head_idx)
        circuit.add_edge(from_node, to_node)

    return circuit

  @staticmethod
  def setup_subparser(subparsers):
    parser = subparsers.add_parser("eap")
    EAPRunner.add_args_to_parser(parser)

  @staticmethod
  def add_args_to_parser(parser):
    add_common_args(parser)

    parser.add_argument("--using-wandb", action="store_true")
    parser.add_argument(
      "--wandb-project", type=str, default="subnetwork-probing"
    )
    parser.add_argument("--wandb-entity", type=str, required=False)
    parser.add_argument("--wandb-group", type=str, required=False)
    parser.add_argument("--wandb-dir", type=str, default="/tmp/wandb")
    parser.add_argument("--wandb-mode", type=str, default="online")
    parser.add_argument(
      "--wandb-run-name",
      type=str,
      required=False,
      default=None,
      help="Value for wandb_run_name",
    )

    parser.add_argument("--top-k", "-k", type=int, default=2, help="Number of edges to keep in the final circuit")

  @classmethod
  def make_default_runner(cls, task: str):
    parser = ArgumentParser()
    cls.add_args_to_parser(parser)
    args = parser.parse_args([])
    return cls(task, args)
