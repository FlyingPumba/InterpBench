import datetime
import os
import pickle
import random
import shutil
from argparse import Namespace
from copy import deepcopy
from typing import Callable, Tuple

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
from circuits_benchmark.utils.ll_model_loader.ll_model_loader_factory import get_ll_model_loader_from_args


class ACDCRunner:
    def __init__(self, case: BenchmarkCase, args: Namespace):
        self.case = case
        self.args = None
        self.configure_acdc(args)

        # Check that dot program is in path
        if not shutil.which("dot"):
          raise ValueError("dot program not in path, cannot generate graphs for ACDC.")

    def configure_acdc(self, _args: Namespace):
        args = deepcopy(_args)
        self.args = args

        if args.torch_num_threads > 0:
          torch.set_num_threads(args.torch_num_threads)

        # Set the seed
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)

        if args.first_cache_cpu is None:
            self.online_cache_cpu = True
        elif args.first_cache_cpu.lower() == "false":
            self.online_cache_cpu = False
        elif args.first_cache_cpu.lower() == "true":
            self.online_cache_cpu = True
        else:
            raise ValueError(f"first_cache_cpu must be either True or False, got {args.first_cache_cpu}")

        if args.second_cache_cpu is None:
            self.corrupted_cache_cpu = True
        elif args.second_cache_cpu.lower() == "false":
            self.corrupted_cache_cpu = False
        elif args.second_cache_cpu.lower() == "true":
            self.corrupted_cache_cpu = True
        else:
            raise ValueError(f"second_cache_cpu must be either True or False, got {args.second_cache_cpu}")

    def run_using_model_loader_from_args(self) -> Tuple[Circuit, CircuitEvalResult]:
      ll_model_loader = get_ll_model_loader_from_args(self.case, self.args)
      clean_dirname = self.prepare_output_dir(ll_model_loader)

      hl_model = self.case.get_hl_model()
      hl_ll_corr, ll_model = ll_model_loader.load_ll_model_and_correspondence(
        load_from_wandb=self.args.load_from_wandb,
        device=self.args.device,
        output_dir=self.args.output_dir,
        same_size=self.args.same_size,
        # IOI specific args:
        eval=True,
        include_mlp=self.args.include_mlp,
        use_pos_embed=self.args.use_pos_embed
      )

      ll_model.eval()
      for param in ll_model.parameters():
          param.requires_grad = False

      # prepare data
      data_size = self.args.data_size
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

      print("Calculating FPR and TPR for threshold", self.args.threshold)
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

      if self.args.using_wandb:
        wandb.init(
          project=f"circuit_discovery{'_same_size' if self.args.same_size else ''}",
          group=f"acdc_{self.case.get_name()}_{str(ll_model_loader.get_output_suffix())}",
          name=f"{self.args.threshold}",
        )
        wandb.save(f"{clean_dirname}/*", base_path=self.args.output_dir)
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

        corrupted_cache_cpu = self.corrupted_cache_cpu
        online_cache_cpu = self.online_cache_cpu

        threshold = self.args.threshold  # only used if >= 0.0
        zero_ablation = True if self.args.zero_ablation else False
        using_wandb = True if self.args.using_wandb else False
        wandb_entity_name = self.args.wandb_entity_name
        wandb_project_name = self.args.wandb_project_name
        wandb_run_name = self.args.wandb_run_name
        wandb_group_name = self.args.wandb_group_name
        indices_mode = self.args.indices_mode
        names_mode = self.args.names_mode
        single_step = True if self.args.single_step else False
        use_pos_embed = self.args.use_pos_embed

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
            wandb_dir=self.args.wandb_dir,
            wandb_mode=self.args.wandb_mode,
            zero_ablation=zero_ablation,
            abs_value_threshold=self.args.abs_value_threshold,
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

        for i in range(self.args.max_num_epochs):
            exp.step(testing=False)

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
      parser = subparsers.add_parser("acdc")
      ACDCRunner.add_args_to_parser(parser)

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
          "--include-mlp", action="store_true", help="Evaluate group 'with_mlp'"
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
      output_suffix = f"{ll_model_loader.get_output_suffix()}/threshold_{self.args.threshold}"
      clean_dirname = f"{self.args.output_dir}/acdc/{self.case.get_name()}/{output_suffix}"

      # remove everything in the directory
      if os.path.exists(clean_dirname):
        shutil.rmtree(clean_dirname)

      # mkdir
      os.makedirs(clean_dirname, exist_ok=True)

      return clean_dirname
