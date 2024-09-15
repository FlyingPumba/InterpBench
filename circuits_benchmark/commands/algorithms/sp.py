import os
import pickle
import random
import shutil
from argparse import Namespace
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from typing import Tuple, Callable, Dict, Optional

import numpy as np
import torch
import wandb
from acdc.TLACDCCorrespondence import TLACDCCorrespondence
from acdc.docstring.utils import AllDataThings
from subnetwork_probing.masked_transformer import CircuitStartingPointType, EdgeLevelMaskedTransformer
from subnetwork_probing.train import NodeLevelMaskedTransformer, iterative_correspondence_from_mask, \
    proportion_of_binary_scores
from transformer_lens import HookedTransformer

from circuits_benchmark.benchmark.benchmark_case import BenchmarkCase
from circuits_benchmark.commands.common_args import add_common_args, add_evaluation_common_ags
from circuits_benchmark.metrics.validation_metrics import l2_metric
from circuits_benchmark.utils.circuit.circuit import Circuit
from circuits_benchmark.utils.circuit.circuit_eval import evaluate_hypothesis_circuit, build_from_acdc_correspondence, \
    CircuitEvalResult
from circuits_benchmark.utils.edge_sp import train_edge_sp, save_edges
from circuits_benchmark.utils.ll_model_loader.ll_model_loader import LLModelLoader
from circuits_benchmark.utils.node_sp import train_sp
from circuits_benchmark.utils.project_paths import get_default_output_dir


@dataclass
class SPConfig:
    seed: Optional[int] = 42
    data_size: Optional[int] = 1000
    device: Optional[str] = "cpu"
    wandb_run_name: Optional[str] = None
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
    wandb_group: Optional[str] = None
    wandb_dir: Optional[str] = None
    wandb_mode: Optional[str] = None
    lr: Optional[float] = 0.01
    epochs: Optional[int] = 300
    verbose: Optional[int] = True
    lambda_reg: Optional[float] = 1
    zero_ablation: Optional[int] = 0
    metric: Optional[str] = "l2"
    edgewise: Optional[bool] = False
    num_examples: Optional[int] = 50
    seq_len: Optional[int] = 300
    n_loss_average_runs: Optional[int] = 4
    torch_num_threads: Optional[int] = 0
    reset_subject: Optional[int] = 0
    print_stats: Optional[int] = 1
    print_every: Optional[int] = 1
    atol: Optional[float] = 1e-1
    use_pos_embed: Optional[bool] = False
    using_wandb: Optional[bool] = False
    output_dir: Optional[str] = get_default_output_dir()
    same_size: Optional[bool] = False

    @staticmethod
    def from_args(args: Namespace) -> "SPConfig":
        return SPConfig(
            seed=int(args.seed),
            data_size=args.data_size,
            device=args.device,
            wandb_run_name=args.wandb_run_name,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
            wandb_group=args.wandb_group,
            wandb_dir=args.wandb_dir,
            wandb_mode=args.wandb_mode,
            lr=args.lr,
            epochs=args.epochs,
            verbose=args.verbose,
            lambda_reg=args.lambda_reg,
            zero_ablation=args.zero_ablation,
            metric=args.metric,
            edgewise=args.edgewise,
            num_examples=args.num_examples,
            seq_len=args.seq_len,
            n_loss_average_runs=args.n_loss_average_runs,
            torch_num_threads=args.torch_num_threads,
            reset_subject=args.reset_subject,
            print_stats=args.print_stats,
            print_every=args.print_every,
            atol=args.atol,
            use_pos_embed=args.use_pos_embed,
            using_wandb=args.using_wandb,
            output_dir=args.output_dir,
            same_size=args.same_size,
        )


class SPRunner:
    def __init__(self,
                 case: BenchmarkCase,
                 config: SPConfig | None = None,
                 args: Namespace | None = None):
        self.case = case
        self.config = config
        self.args = deepcopy(args)

        if self.config is None:
            self.config = SPConfig.from_args(args)

        assert self.config is not None
        self.configure_sp()

        # Check that dot program is in path
        if not shutil.which("dot"):
            raise ValueError("dot program not in path, cannot generate graphs for ACDC.")

    def configure_sp(self):
        if self.config.torch_num_threads > 0:
            torch.set_num_threads(self.config.torch_num_threads)

        # Set the seed
        torch.manual_seed(self.config.seed)
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)

    def eval_fn(self, corr: TLACDCCorrespondence):
        sp_circuit = build_from_acdc_correspondence(corr=corr)
        return evaluate_hypothesis_circuit(
            sp_circuit,
            self.ll_model,
            self.hl_ll_corr,
            self.case,
            print_summary=False,
        )

    def run_using_model_loader(self, ll_model_loader: LLModelLoader) -> Tuple[Circuit, CircuitEvalResult]:
        clean_dirname = self.prepare_output_dir(ll_model_loader)

        hl_ll_corr, ll_model = ll_model_loader.load_ll_model_and_correspondence(
            device=self.config.device,
            output_dir=self.config.output_dir,
            same_size=self.config.same_size,
            # IOI specific args:
            use_pos_embed=self.config.use_pos_embed
        )
        self.ll_model = ll_model
        self.hl_ll_corr = hl_ll_corr

        ll_model.eval()
        for param in ll_model.parameters():
            param.requires_grad = False

        images_output_dir = os.path.join(clean_dirname, "images")
        os.makedirs(images_output_dir, exist_ok=True)

        metric_name = self.config.metric

        data_size = self.config.data_size
        clean_data = self.case.get_clean_data(max_samples=int(1.2 * data_size))
        corrupted_data = self.case.get_corrupted_data(max_samples=int(1.2 * data_size))

        clean_outputs = clean_data.get_targets()
        baseline_output = clean_outputs[:data_size]
        test_baseline_output = clean_outputs[data_size:]

        if isinstance(clean_outputs, list):
            clean_outputs = torch.stack(clean_outputs, dim=0)
            baseline_output = torch.stack(baseline_output, dim=0)
            test_baseline_output = torch.stack(test_baseline_output, dim=0)

        baseline_output = baseline_output.to(self.config.device)
        test_baseline_output = test_baseline_output.to(self.config.device)

        if metric_name == "l2":
            validation_metric = partial(
                l2_metric,
                baseline_output=baseline_output,
                is_categorical=self.case.is_categorical(),
            )
            test_loss_metric = partial(
                l2_metric,
                baseline_output=test_baseline_output,
                is_categorical=self.case.is_categorical(),
            )
            test_accuracy_fn = (
                lambda x, y: torch.isclose(x, y, atol=self.config.atol).float().mean()
            )
            test_accuracy_metric = partial(test_accuracy_fn, test_baseline_output)
        elif metric_name == "kl":
            kl_metric = (
                lambda x, y: torch.nn.functional.kl_div(
                    torch.nn.functional.log_softmax(x, dim=-1),
                    torch.nn.functional.softmax(y, dim=-1),
                    reduction="none",
                )
                .sum(dim=-1)
                .mean()
            )

            validation_metric = partial(kl_metric, y=baseline_output)
            test_loss_metric = partial(kl_metric, y=test_baseline_output)
            test_accuracy_fn = (
                lambda x, y: (x.argmax(dim=-1) == y.argmax(dim=-1)).float().mean()
            )
            test_accuracy_metric = partial(test_accuracy_fn, test_baseline_output)
        else:
            raise NotImplementedError(f"Metric {metric_name} not implemented")
        test_metrics = {"loss": test_loss_metric, "accuracy": test_accuracy_metric}

        sp_circuit, log_dict = self.run(
            ll_model,
            validation_metric,
            clean_data.get_inputs(),
            clean_outputs,
            corrupted_data.get_inputs(),
            test_metrics,
            clean_dirname,
        )

        print("Calculating FPR and TPR for regularizer", self.config.lambda_reg)
        result = evaluate_hypothesis_circuit(
            sp_circuit,
            ll_model,
            hl_ll_corr,
            self.case,
        )
        # save results
        pickle.dump(result, open(f"{clean_dirname}/result.pkl", "wb"))

        if self.config.using_wandb:
            wandb.log(
                {
                    "regularizer": self.config.lambda_reg,
                    "nodes_fpr": result.nodes.fpr,
                    "nodes_tpr": result.nodes.tpr,
                    "edges_fpr": result.edges.fpr,
                    "edges_tpr": result.edges.tpr,
                    "percentage_binary": log_dict["percentage_binary"],
                }
            )
            wandb.finish()
            wandb.init(
                project=f"circuit_discovery{'_same_size' if self.config.same_size else ''}",
                group=f"{'edge' if self.config.edgewise else 'node'}_sp_{self.case.get_name()}_{ll_model_loader.get_output_suffix()}",
                name=f"{self.config.lambda_reg}",
            )
            wandb.save(f"{clean_dirname}/*", base_path=self.config.output_dir)

        return sp_circuit, result

    def run(
        self,
        tl_model: HookedTransformer,
        validation_metric: Callable[[torch.Tensor], torch.Tensor],
        clean_inputs: torch.Tensor,
        clean_outputs: torch.Tensor,
        corrupted_inputs: torch.Tensor,
        test_metrics: Dict[str, Callable[[torch.Tensor], torch.Tensor]],
        output_dir: str
    ) -> Tuple[Circuit, dict]:
        zero_ablation = True if self.config.zero_ablation else False
        use_pos_embed = True
        edgewise = self.config.edgewise
        data_size = self.config.data_size

        clean_inputs = clean_inputs.to(self.config.device)
        corrupted_inputs = corrupted_inputs.to(self.config.device)
        clean_outputs = clean_outputs.to(self.config.device)

        all_task_things = AllDataThings(
            tl_model=tl_model,
            validation_metric=validation_metric,
            validation_data=clean_inputs[:data_size],
            validation_labels=clean_outputs[:data_size],
            validation_mask=None,
            validation_patch_data=corrupted_inputs[:data_size],
            test_metrics=test_metrics,
            test_data=clean_inputs[data_size:],
            test_labels=clean_outputs[data_size:],
            test_mask=None,
            test_patch_data=corrupted_inputs[data_size:],
        )

        # Setup wandb if needed
        if self.config.wandb_run_name is None:
            self.config.wandb_run_name = f"SP_{'edge' if edgewise else 'node'}_{self.case.get_name()}_reg_{self.config.lambda_reg}{'_zero' if zero_ablation else ''}"
        self.config.wandb_name = self.config.wandb_run_name

        # prepare Masked Model
        tl_model.reset_hooks()
        if edgewise:
            masked_model = EdgeLevelMaskedTransformer(
                tl_model,
                starting_point_type=(
                    CircuitStartingPointType.RESID_PRE
                    if not use_pos_embed
                    else CircuitStartingPointType.POS_EMBED
                ),
            )
        else:
            masked_model = NodeLevelMaskedTransformer(tl_model)
        masked_model = masked_model.to(self.config.device)

        # Run SP
        masked_model.freeze_weights()
        if edgewise:
            print(f"Running Edgewise SP with lambda_reg={self.config.lambda_reg}")
            masked_model, log_dict = train_edge_sp(
                args=self.config,
                masked_model=masked_model,
                all_task_things=all_task_things,
                print_every=self.config.print_every,
                eval_fn=self.eval_fn,
            )
            percentage_binary = masked_model.proportion_of_binary_scores()
            sp_corr = masked_model.get_edge_level_correspondence_from_masks(
                use_pos_embed=use_pos_embed
            )
        else:
            print(f"Running Node SP with lambda_reg={self.config.lambda_reg}")
            masked_model, log_dict = train_sp(
                args=self.config,
                masked_model=masked_model,
                all_task_things=all_task_things,
            )
            percentage_binary = proportion_of_binary_scores(masked_model)
            sp_corr, _ = iterative_correspondence_from_mask(
                masked_model.model,
                log_dict["nodes_to_mask"],
                use_pos_embed=use_pos_embed,
            )

        # Update dict with some different things
        log_dict["percentage_binary"] = percentage_binary

        # save sp circuit edges
        save_edges(sp_corr, f"{output_dir}/edges.pkl")

        # Build and return circuit
        sp_circuit = build_from_acdc_correspondence(corr=sp_corr)
        return sp_circuit, log_dict

    @staticmethod
    def setup_subparser(subparsers):
        parser = subparsers.add_parser("sp")
        SPRunner.add_args_to_parser(parser)

    @staticmethod
    def add_args_to_parser(parser):
        add_common_args(parser)
        add_evaluation_common_ags(parser)

        parser.add_argument("--using_wandb", action="store_true")
        parser.add_argument("--wandb-project", type=str, default="subnetwork-probing")
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
        parser.add_argument("--lr", type=float, default=0.01)
        parser.add_argument("--epochs", type=int, default=300)
        parser.add_argument("--verbose", type=int, default=1)
        parser.add_argument("--lambda-reg", type=float, default=1)
        parser.add_argument("--zero-ablation", type=int, default=0)
        parser.add_argument("--data-size", type=int, default=1000)
        parser.add_argument("--metric", type=str, choices=["l2", "kl"], default="l2")
        parser.add_argument("--edgewise", action="store_true")
        parser.add_argument("--num-examples", type=int, default=50)
        parser.add_argument("--seq-len", type=int, default=300)
        parser.add_argument("--n-loss-average-runs", type=int, default=4)
        parser.add_argument(
            "--torch-num-threads",
            type=int,
            default=0,
            help="How many threads to use for torch (0=all)",
        )
        parser.add_argument("--reset-subject", type=int, default=0)
        # parser.add_argument("--torch-num-threads", type=int, default=0)
        parser.add_argument("--print-stats", type=int, default=1, required=False)
        parser.add_argument("--print-every", type=int, default=1, required=False)
        parser.add_argument("--atol", type=float, default=5e-2, required=False)
        parser.add_argument(
            "--use-pos-embed", action="store_true", help="Use positional embeddings"
        )

    def prepare_output_dir(self, ll_model_loader) -> str:
        clean_dirname = os.path.join(
            self.config.output_dir,
            f"{'edge_' if self.config.edgewise else 'node_'}sp/{self.case.get_name()}",
            ll_model_loader.get_output_suffix(),
            f"lambda_{self.config.lambda_reg}",
        )

        # remove everything in the directory
        if os.path.exists(clean_dirname):
            shutil.rmtree(clean_dirname)

        # mkdir
        os.makedirs(clean_dirname, exist_ok=True)

        return str(clean_dirname)
