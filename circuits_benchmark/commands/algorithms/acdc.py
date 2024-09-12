import os
import pickle
import random
import shutil
from argparse import Namespace
from copy import deepcopy
from typing import Tuple, Literal

import numpy as np
import torch as t
import wandb
from auto_circuit.data import PromptDataset, PromptDataLoader
from auto_circuit.prune_algos.ACDC import acdc_prune_scores
from auto_circuit.types import PruneScores, OutputSlice
from auto_circuit.utils.graph_utils import patchable_model

from circuits_benchmark.benchmark.benchmark_case import BenchmarkCase
from circuits_benchmark.commands.algorithms.legacy_acdc import ACDCConfig, LegacyACDCRunner
from circuits_benchmark.commands.common_args import add_common_args, add_evaluation_common_ags
from circuits_benchmark.utils.auto_circuit_utils import build_circuit
from circuits_benchmark.utils.circuit.circuit import Circuit
from circuits_benchmark.utils.circuit.circuit_eval import evaluate_hypothesis_circuit, CircuitEvalResult
from circuits_benchmark.utils.ll_model_loader.ll_model_loader import LLModelLoader


class ACDCRunner:
    def __init__(self,
                 case: BenchmarkCase,
                 config: ACDCConfig | None = None,
                 args: Namespace | None = None):
        self.case = case
        self.config = config
        self.args = deepcopy(args)

        if self.config is None:
            self.config = ACDCConfig.from_args(args)

        assert self.config is not None
        self.configure_acdc()

    def configure_acdc(self):
        if self.config.torch_num_threads > 0:
            t.set_num_threads(self.config.torch_num_threads)

        # Set the seed
        t.manual_seed(self.config.seed)
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)

    def run_using_model_loader(self, ll_model_loader: LLModelLoader) -> Tuple[Circuit, CircuitEvalResult]:
        clean_dirname = self.prepare_output_dir(ll_model_loader)

        print(f"Running ACDC evaluation for case {self.case.get_name()} ({str(ll_model_loader)})")
        print(f"Output directory: {clean_dirname}")

        hl_ll_corr, ll_model = ll_model_loader.load_ll_model_and_correspondence(
            device=self.config.device,
            output_dir=self.config.output_dir,
            same_size=self.config.same_size,
            # IOI specific args:
            use_pos_embed=self.config.use_pos_embed
        )
        hl_model = self.case.get_hl_model()

        ll_model.eval()
        for param in ll_model.parameters():
            param.requires_grad = False

        # prepare data
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

        faithfulness_metric: Literal["kl_div", "mse"] = "mse" if not hl_model.is_categorical() else "kl_div"

        acdc_circuit = self.run(
            ll_model,
            clean_dataset.get_inputs(),
            clean_outputs,
            corrupted_dataset.get_inputs(),
            corrupted_outputs,
            faithfulness_metric,
        )

        print("Done running acdc: ")
        print(list(acdc_circuit.nodes), list(acdc_circuit.edges))

        print("hl_ll_corr:", hl_ll_corr)
        hl_ll_corr.save(f"{clean_dirname}/hl_ll_corr.pkl")

        print("Calculating FPR and TPR for threshold", self.config.threshold)
        gt_circuit = None
        if str(ll_model_loader) == "ground_truth":
            gt_circuit = self.case.get_hl_gt_circuit(granularity="acdc_hooks")

        result = evaluate_hypothesis_circuit(
            acdc_circuit,
            ll_model,
            hl_ll_corr,
            self.case,
            gt_circuit=gt_circuit,
        )

        # save the result
        with open(f"{clean_dirname}/result.txt", "w") as f:
            f.write(str(result))
        pickle.dump(result, open(f"{clean_dirname}/result.pkl", "wb"))
        print(f"Saved result to {clean_dirname}/result.txt and {clean_dirname}/result.pkl")

        if self.config.using_wandb:
            wandb.init(
                project=f"circuit_discovery{'_same_size' if self.config.same_size else ''}",
                group=f"acdc_{self.case.get_name()}_{str(ll_model_loader.get_output_suffix())}",
                name=f"{self.config.threshold}",
            )
            wandb.save(f"{clean_dirname}/*", base_path=self.config.output_dir)
            wandb.finish()

        return acdc_circuit, result

    def run(
        self,
        tl_model: t.nn.Module,
        clean_inputs: t.Tensor,
        clean_outputs: t.Tensor,
        corrupted_inputs: t.Tensor,
        corrupted_outputs: t.Tensor,
        faithfulness_metric: Literal["kl_div", "mse"],
    ) -> Circuit:
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
        train_loader = PromptDataLoader(dataset,
                                        seq_len=self.case.get_max_seq_len(),
                                        diverge_idx=0,
                                        batch_size=len(dataset))

        attribution_scores: PruneScores = acdc_prune_scores(
            model=auto_circuit_model,
            dataloader=train_loader,
            official_edges=None,
            tao_exps=[0],  # i.e., threshold * (10**0) = threshold
            tao_bases=[self.config.threshold],  # type: ignore
            faithfulness_target=faithfulness_metric,
        )

        acdc_circuit = build_circuit(auto_circuit_model, attribution_scores, self.config.threshold)
        acdc_circuit.save(f"{self.config.output_dir}/final_circuit.pkl")

        return acdc_circuit

    @staticmethod
    def add_args_to_parser(parser):
        add_common_args(parser)
        add_evaluation_common_ags(parser)

        parser.add_argument(
            "-t",
            "--threshold",
            type=float,
            default=0.025,
            help="ACDC's threshold for pruning edges",
        )
        parser.add_argument(
            "--data-size",
            type=int,
            required=False,
            default=1000,
            help="How many samples to use"
        )
        parser.add_argument(
            "--next-token", action="store_true", help="Use next token model"
        )
        parser.add_argument(
            "--use-pos-embed", action="store_true", help="Use positional embeddings"
        )
        parser.add_argument("--indices-mode", type=str, default="normal")
        parser.add_argument("--names-mode", type=str, default="normal")
        parser.add_argument("--torch-num-threads", type=int, default=0,
                            help="How many threads to use for torch (0=all)")
        parser.add_argument("--max-num-epochs", type=int, default=100_000)
        parser.add_argument("--single-step", action="store_true", help="Use single step, mostly for testing")
        parser.add_argument(
            "--abs-value-threshold", action="store_true", help="Use the absolute value of the result to check threshold"
        )
        parser.add_argument(
            "--first-cache-cpu",
            type=str,
            required=False,
            default="True",
            help="Value for first_cache_cpu (the old name for the `online_cache`)",
        )
        parser.add_argument(
            "--second-cache-cpu",
            type=str,
            required=False,
            default="True",
            help="Value for second_cache_cpu (the old name for the `corrupted_cache`)",
        )
        parser.add_argument("--zero-ablation", action="store_true", help="Use zero ablation")

        parser.add_argument(
            "-wandb", "--using_wandb", action="store_true", help="Use wandb"
        )
        parser.add_argument(
            "--wandb-entity-name",
            type=str,
            required=False,
            default="remix_school-of-rock",
            help="Value for wandb_entity_name",
        )
        parser.add_argument(
            "--wandb-group-name", type=str, required=False, default="default", help="Value for wandb_group_name"
        )
        parser.add_argument(
            "--wandb-project-name", type=str, required=False, default="acdc", help="Value for wandb_project_name"
        )
        parser.add_argument("--wandb-run-name", type=str, required=False, default=None, help="Value for wandb_run_name")
        parser.add_argument("--wandb-dir", type=str, default="/tmp/wandb")
        parser.add_argument("--wandb-mode", type=str, default="online")

    @staticmethod
    def setup_subparser(subparsers):
        parser = subparsers.add_parser("acdc")
        LegacyACDCRunner.add_args_to_parser(parser)

    def prepare_output_dir(self, ll_model_loader):
        output_suffix = f"{ll_model_loader.get_output_suffix()}/threshold_{self.config.threshold}"
        clean_dirname = f"{self.config.output_dir}/acdc/{self.case.get_name()}/{output_suffix}"

        # remove everything in the directory
        if os.path.exists(clean_dirname):
            shutil.rmtree(clean_dirname)

        # mkdir
        os.makedirs(clean_dirname, exist_ok=True)

        return clean_dirname
