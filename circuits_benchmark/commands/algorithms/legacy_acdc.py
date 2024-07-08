import datetime
import os
import pickle
import random
import shutil
from argparse import Namespace
from copy import deepcopy
from dataclasses import dataclass
from typing import Callable, Tuple, Optional, Literal

import numpy as np
import torch
import wandb
from acdc.TLACDCExperiment import TLACDCExperiment
from acdc.acdc_graphics import show
from transformer_lens import HookedTransformer

from circuits_benchmark.benchmark.benchmark_case import BenchmarkCase
from circuits_benchmark.commands.common_args import add_common_args, add_evaluation_common_ags
from circuits_benchmark.utils.circuit.circuit import Circuit
from circuits_benchmark.utils.circuit.circuit_eval import build_from_acdc_correspondence, evaluate_hypothesis_circuit, \
  CircuitEvalResult
from circuits_benchmark.utils.ll_model_loader.ll_model_loader import LLModelLoader
from circuits_benchmark.utils.project_paths import get_default_output_dir


@dataclass
class ACDCConfig:
    threshold: Optional[float] = 0.025
    data_size: Optional[int] = 1000
    include_mlp: Optional[bool] = False
    next_token: Optional[bool] = False
    use_pos_embed: Optional[bool] = False
    indices_mode: Literal["normal", "reverse", "shuffle"] = "reverse",
    names_mode: Literal["normal", "reverse", "shuffle"] = "normal",
    torch_num_threads: Optional[int] = 0
    max_num_epochs: Optional[int] = 100_000
    single_step: Optional[bool] = False
    abs_value_threshold: Optional[bool] = False
    online_cache_cpu: Optional[bool] = True
    corrupted_cache_cpu: Optional[bool] = True
    zero_ablation: Optional[bool] = False
    using_wandb: Optional[bool] = False
    wandb_entity_name: Optional[str] = None
    wandb_group_name: Optional[str] = None
    wandb_project_name: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_dir: Optional[str] = "/tmp/wandb"
    wandb_mode: Optional[str] = "online"
    seed: Optional[int] = 42
    output_dir: Optional[str] = get_default_output_dir()
    same_size: Optional[bool] = False
    device: Optional[str] = "cpu"
    testing: Optional[bool] = False

    @staticmethod
    def from_args(args: Namespace) -> "ACDCConfig":
        config = ACDCConfig(
            threshold=args.threshold,
            seed=int(args.seed),
            data_size=args.data_size,
            include_mlp=args.include_mlp,
            next_token=args.next_token,
            use_pos_embed=args.use_pos_embed,
            indices_mode=args.indices_mode,
            names_mode=args.names_mode,
            torch_num_threads=args.torch_num_threads,
            max_num_epochs=args.max_num_epochs,
            single_step=args.single_step,
            abs_value_threshold=args.abs_value_threshold,
            online_cache_cpu=True,
            corrupted_cache_cpu=True,
            zero_ablation=args.zero_ablation,
            using_wandb=args.using_wandb,
            wandb_entity_name=args.wandb_entity_name,
            wandb_group_name=args.wandb_group_name,
            wandb_project_name=args.wandb_project_name,
            wandb_run_name=args.wandb_run_name,
            wandb_dir=args.wand,
            wandb_mode=args.wandb_mode,
            output_dir=args.output_dir,
            same_size=args.same_size,
            device=args.device,
        )

        if args.first_cache_cpu is None:
          config.online_cache_cpu = True
        elif args.first_cache_cpu.lower() == "false":
          config.online_cache_cpu = False
        elif args.first_cache_cpu.lower() == "true":
          config.online_cache_cpu = True
        else:
          raise ValueError(f"first_cache_cpu must be either True or False, got {args.first_cache_cpu}")

        if args.config.second_cache_cpu is None:
          config.corrupted_cache_cpu = True
        elif args.config.second_cache_cpu.lower() == "false":
          config.corrupted_cache_cpu = False
        elif args.config.second_cache_cpu.lower() == "true":
          config.corrupted_cache_cpu = True
        else:
          raise ValueError(f"second_cache_cpu must be either True or False, got {args.config.second_cache_cpu}")

        return config

class LegacyACDCRunner:
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
        torch.set_num_threads(self.config.torch_num_threads)

      # Set the seed
      torch.manual_seed(self.config.seed)
      random.seed(self.config.seed)
      np.random.seed(self.config.seed)

      # Check that dot program is in path
      if not shutil.which("dot"):
        raise ValueError("dot program not in path, cannot generate graphs for ACDC.")

    def run_using_model_loader(self, ll_model_loader: LLModelLoader) -> Tuple[Circuit, CircuitEvalResult]:
      clean_dirname = self.prepare_output_dir(ll_model_loader)

      hl_ll_corr, ll_model = ll_model_loader.load_ll_model_and_correspondence(
        device=self.config.device,
        output_dir=self.config.output_dir,
        same_size=self.config.same_size,
        # IOI specific args:
        eval=True,
        include_mlp=self.config.include_mlp,
        use_pos_embed=self.config.use_pos_embed
      )
      hl_model = self.case.get_hl_model()

      ll_model.eval()
      for param in ll_model.parameters():
          param.requires_grad = False

      # prepare data
      data_size = self.config.data_size
      clean_data = self.case.get_clean_data(max_samples=data_size).get_inputs()
      corrupted_data = self.case.get_corrupted_data(max_samples=data_size).get_inputs()

      # prepare validation metric
      metric_name = "l2" if not hl_model.is_categorical() else "kl"
      validation_metric = self.case.get_validation_metric(ll_model, metric_name=metric_name, data=clean_data)

      acdc_circuit = self.run(
        ll_model,
        clean_data,
        corrupted_data,
        validation_metric,
        clean_dirname,
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
        tl_model: HookedTransformer,
        clean_dataset: torch.Tensor,
        corrupt_dataset: torch.Tensor,
        validation_metric: Callable[[torch.Tensor], torch.Tensor],
        output_dir: str,
    ) -> Circuit:
        images_output_dir = os.path.join(output_dir, "images")
        os.makedirs(images_output_dir, exist_ok=True)

        corrupted_cache_cpu = self.config.corrupted_cache_cpu
        online_cache_cpu = self.config.online_cache_cpu

        threshold = self.config.threshold  # only used if >= 0.0
        zero_ablation = True if self.config.zero_ablation else False
        using_wandb = True if self.config.using_wandb else False
        wandb_entity_name = self.config.wandb_entity_name
        wandb_project_name = self.config.wandb_project_name
        wandb_run_name = self.config.wandb_run_name
        wandb_group_name = self.config.wandb_group_name
        indices_mode = self.config.indices_mode
        names_mode = self.config.names_mode
        single_step = True if self.config.single_step else False
        use_pos_embed = self.config.use_pos_embed

        tl_model.reset_hooks()
        exp = TLACDCExperiment(
            model=tl_model,
            threshold=threshold,
            images_output_dir=images_output_dir,
            using_wandb=using_wandb,
            wandb_entity_name=wandb_entity_name,
            wandb_project_name=wandb_project_name,
            wandb_run_name=wandb_run_name,
            wandb_group_name=wandb_group_name,
            wandb_dir=self.config.wandb_dir,
            wandb_mode=self.config.wandb_mode,
            zero_ablation=zero_ablation,
            abs_value_threshold=self.config.abs_value_threshold,
            ds=clean_dataset,
            ref_ds=corrupt_dataset,
            metric=validation_metric,
            second_metric=None,
            verbose=True,
            indices_mode=indices_mode,
            names_mode=names_mode,
            corrupted_cache_cpu=corrupted_cache_cpu,
            hook_verbose=False,
            online_cache_cpu=online_cache_cpu,
            add_sender_hooks=True,
            use_pos_embed=use_pos_embed,
            add_receiver_hooks=False,
            remove_redundant=False,
            show_full_index=use_pos_embed,
        )

        exp_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        for i in range(self.config.max_num_epochs):
            exp.step(testing=self.config.testing)

            show(
                exp.corr,
                fname=f"{images_output_dir}/img_new_{i + 1}.png",
            )

            print(i, "-" * 50)
            print(exp.count_num_edges())

            if i == 0:
                exp.save_edges(os.path.join(output_dir, "edges.pkl"))

            if exp.current_node is None or single_step:
                show(
                    exp.corr,
                    fname=f"{images_output_dir}/ACDC_new_{exp_time}.png",
                    show_placeholders=True,
                )
                break

        exp.save_edges(os.path.join(output_dir, "another_final_edges.pkl"))

        exp.save_subgraph(
            fpath=f"{output_dir}/subgraph.pth",
            return_it=True,
        )

        acdc_circuit = build_from_acdc_correspondence(exp.corr)
        acdc_circuit.save(f"{output_dir}/final_circuit.pkl")

        return acdc_circuit

    @staticmethod
    def setup_subparser(subparsers):
      parser = subparsers.add_parser("legacy_acdc")
      LegacyACDCRunner.add_args_to_parser(parser)

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
          "--include-mlp", type=int, help="Evaluate group 'with_mlp'", default=1
      )
      parser.add_argument(
          "--next-token", action="store_true", help="Use next token model"
      )
      parser.add_argument(
          "--use-pos-embed", action="store_true", help="Use positional embeddings"
      )
      parser.add_argument("--indices-mode", type=str, default="normal")
      parser.add_argument("--names-mode", type=str, default="normal")
      parser.add_argument("--torch-num-threads", type=int, default=0, help="How many threads to use for torch (0=all)")
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

    def prepare_output_dir(self, ll_model_loader):
      output_suffix = f"{ll_model_loader.get_output_suffix()}/threshold_{self.config.threshold}"
      clean_dirname = f"{self.config.output_dir}/legacy_acdc/{self.case.get_name()}/{output_suffix}"

      # remove everything in the directory
      if os.path.exists(clean_dirname):
        shutil.rmtree(clean_dirname)

      # mkdir
      os.makedirs(clean_dirname, exist_ok=True)

      return clean_dirname
